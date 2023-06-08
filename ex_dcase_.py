from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import argparse
import torch.nn.functional as F
from codecarbon import EmissionsTracker
from datasets.audiodataset import get_test_set, get_val_set, get_training_set
from models.selectModel import get_model
from models.mel import AugmentMelSTFT
from models.teacher_mel import AugmentMelSTFT as AugmentTeacherMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import mixup, mixstyle, spawn_get
from helpers.lr_schedule import exp_warmup_linear_down
import time
import urllib.parse
from torch.hub import load_state_dict_from_url
from models.MobileNetV3 import get_model as get_mobilenet
from models.passt.passt import get_model as get_passt
from helpers.utils import NAME_TO_WIDTH
import numpy as np
import random2

# points to github releases
model_url = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
# folder to store downloaded models to
model_dir = "teacher_models"

use_float16 = False

class SimpleDCASELitModule(pl.LightningModule):
    """
    This is a Pytorch Lightening Module.
    It has several convenient abstractions, e.g. we don't have to specify all parts of the
    training loop (optimizer.step(), loss.backward()) ourselves.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse and contains all configurations for our experiment
        # model to preprocess waveforms into log mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=config.n_mels,
                                  sr=config.resample_rate,
                                  win_length=config.window_size,
                                  hopsize=config.hop_size,
                                  n_fft=config.n_fft,
                                  freqm=config.freqm,
                                  timem=config.timem,
                                  fmin=config.fmin,
                                  fmax=config.fmax,
                                  fmin_aug_range=config.fmin_aug_range,
                                  fmax_aug_range=config.fmax_aug_range
                                  )
        if config.teacher_model is not None:
            self.teacher_mel = AugmentTeacherMelSTFT(n_mels=config.n_mels,
                                  sr=config.resample_rate,
                                  win_length=config.window_size,
                                  hopsize=config.hop_size,
                                  n_fft=config.n_fft,
                                  freqm=config.freqm,
                                  timem=config.timem,
                                  fmin=config.fmin,
                                  fmax=config.fmax,
                                  fmin_aug_range=config.fmin_aug_range,
                                  fmax_aug_range=config.fmax_aug_range
                                  )

        # our model to be trained on the log mel spectrograms
        #self.model = get_model( modelarch=config.architecture,
        #                        in_channels=config.in_channels,
        #                        n_classes=config.n_classes,
        #                        base_channels=config.base_channels,
        #                        channels_multiplier=config.channels_multiplier
        #                       )

        # quantization
        self.quant = config.quant

        # mixstyle parameters
        self.mixstyle_p = config.mixstyle_p
        self.mixstyle_alpha = config.mixstyle_alpha

        # define knowledge distillation parameters
        self.temperature = config.temperature
        self.soft_targets_weight = config.soft_targets_weight  # weight loss for soft targets
        self.kl_div_loss = nn.KLDivLoss(log_target=True, reduction="none")  # KL Divergence loss for soft targets
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # define parameters for teacher
        if config.teacher_model is not None:
            self.teacher_net = get_teacher_model(config, config.teacher_model)

        if use_float16:
            self.model = get_model(config).to(torch.float16)
        else:
            self.model = get_model(config)

    def mel_forward(self, x):
        """
        @param x: a batch of raw signals (waveform)
        return: a batch of log mel spectrograms
        """
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])  # for calculating mel spectrograms we remove the channel dimension

        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])  # batch x channels x mels x time-frames
        return x

    # calculates only the mel spectogram of the input for the teacher
    def teacher_mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.teacher_mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        if use_float16:
            x = self.mel_forward(x).to(torch.float16)
            x = self.model(x).to(torch.float16)
        else:
            x = self.mel_forward(x)
            x = self.model(x)

        return x

    # forward pass of the teacher
    def teacher_forward(self, x):
        self.teacher_net.eval()
        return self.teacher_net(x)

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        schedule_lambda = \
            exp_warmup_linear_down(self.config.warm_up_len, self.config.ramp_down_len, self.config.ramp_down_start,
                                   self.config.last_lr_value)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx: will likely not be used at all
        :return: a dict containing at least loss that is used to update model parameters, can also contain
                    other items that can be processed in 'training_epoch_end' to log other metrics than loss
        """
        x, y = train_batch  # we get a batch of raw audio signals and labels as defined by our dataset
        bs = x.size(0)
        batch_size = len(y)




        # create teacher spectrogram
        if self.config.teacher_model is not None:
            x_teacher = self.teacher_mel_forward(x)

        if use_float16:
            x = x.to(torch.float16)

        # create student spectrogram
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        if use_float16:
            x = x.to(torch.float16)

        if self.mixstyle_p > 0:
            x = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha)
            x_teacher = mixstyle(x_teacher, self.mixstyle_p, self.mixstyle_alpha)

        if args.mixup_alpha:
            # Apply Mixup, a very common data augmentation method
            rn_indices, lam = mixup(bs, args.mixup_alpha)  # get shuffled indices and mixing coefficients
            # send mixing coefficients to correct device and make them 4-dimensional
            lam = lam.to(x.device).reshape(bs, 1, 1, 1)
            # mix two spectrograms from the batch
            x = x * lam + x[rn_indices] * (1. - lam)

            if self.config.teacher_model is not None:
                # applying same mixup config also to teacher spectrograms
                x_teacher = x_teacher * lam.reshape(batch_size, 1, 1, 1) + \
                            x_teacher[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))

            if use_float16:
                x = x.to(torch.float16)
            # generate predictions for mixed log mel spectrograms
            y_hat = self.model(x)
            # mix the prediction targets using the same mixing coefficients
            samples_loss = (
                    F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                    F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(bs))
            )

        else:
            if use_float16:
                x = x.to(torch.float16)
            y_hat = self.model(x)
            # cross_entropy is used for multiclass problems
            # be careful when choosing the correct loss functions
            # read the documentation what input your loss function expects, e.g. for F.cross_entropy:
            # the logits (no softmax!) and the prediction targets (class indices)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        label_loss = samples_loss.mean()
        results = {"loss": label_loss}
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred = (preds == y).sum()

        # getting teacher predictions
        if self.config.teacher_model is not None:
            with torch.no_grad():
                # inference step using teacher ensemble
                y_hat_teacher = self.teacher_forward(x_teacher)

        # Temperature adjusted probabilities of teacher and student
        if self.config.teacher_model is not None:
            with torch.cuda.amp.autocast():
                y_soft_teacher = self.log_softmax(y_hat_teacher / self.temperature)
                y_soft_student = self.log_softmax(y_hat / self.temperature)

        # distillation loss
        if self.config.teacher_model is not None:
            soft_targets_loss = self.kl_div_loss(y_soft_student, y_soft_teacher).mean()  # mean since reduction="none"

        if self.config.teacher_model is not None:
            loss = self.soft_targets_weight * soft_targets_loss + label_loss
        else:
            loss = label_loss
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # logging results for losses and accuracies

        if self.config.teacher_model is not None:
            results = {"loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y),
                   "teacher_loss_weighted": soft_targets_loss.detach() * self.soft_targets_weight,
                   "label_loss": label_loss.detach()}
        else:
            results = {"loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y),
                       "label_loss": label_loss.detach()}

        return results

    def training_epoch_end(self, outputs):
        """
        :param outputs: contains the items you log in 'training_step'
        :return: a dict containing the metrics you want to log to Weights and Biases
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        train_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)

        logs = {'train.loss': avg_loss, 'train_acc': train_acc, 'step': self.current_epoch}
        avg_label_loss = torch.stack([x['label_loss'] for x in outputs]).mean()

        if self.config.teacher_model is not None:
            avg_teacher_loss_weighted = torch.stack([x['teacher_loss_weighted'] for x in outputs]).mean()
            logs['teacher_loss_weighted'] = avg_teacher_loss_weighted

        logs['label_loss'] = avg_label_loss

        self.log_dict(logs)
        #self.log_dict({'loss': avg_loss})

    def validation_step(self, val_batch, batch_idx):
        # similar to 'training_step' but without any data augmentation
        # pytorch lightening takes care of 'with torch.no_grad()' and 'model.eval()'
        x, y = val_batch

        if use_float16:
            x = x.to(torch.float16)

        x = self.mel_forward(x)

        if use_float16:
            x = x.to(torch.float16)

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # log validation metric to weights and biases
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log_dict({'val_loss': avg_loss})

    def test_step(self, test_batch, batch_idx):
        # the test step is used to evaluate the quantized model
        x, y = test_batch

        if use_float16:
            x = x.to(torch.float16)


        x = self.mel_forward(x)

        if use_float16:
            x = x.to(torch.float16)

        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, y, reduction="none")
        loss = samples_loss.mean()

        _, preds = torch.max(y_hat, dim=1)
        results = {"preds": preds, "labels": y, "test_loss": loss}
        return results

    def test_epoch_end(self, outputs):
        #c_names = self.trainer.test_dataloaders[0].dataset.dataset.c_names  # class names for reporting acc
        c_names = ["airport", "shopping mall", "metro station", "pedestrian street", "public square",
                   "street traffic", "tram", "bus", "metro", "park"]
        c_names = ["acc_"+x for x in c_names]

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        metric = Accuracy(task="multiclass", num_classes=len(c_names), average="none").to(device)  # setup class wise acc
        # stack output from batches
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs], dim=0)
        labels = torch.cat([x["labels"] for x in outputs], dim=0)

        # cm = torch.Tensor(confusion_matrix(labels, preds))
        acc = metric(preds, labels)
        logs = {'test_loss': avg_loss, 'mac_acc': acc.mean(), 'step': self.current_epoch}
        logs.update(dict(zip(c_names, acc)))
        self.log_dict(logs)


def get_teacher_model(config, teacher_model):
    if teacher_model == "mobilenet":
        pretrained_name = "mn40_as_no_im_pre"
        if pretrained_name:
            model = get_mobilenet(width_mult=NAME_TO_WIDTH(pretrained_name), pretrained_name=pretrained_name,
                                  head_type=config.head_type, se_dims=config.se_dims, num_classes=10)
        else:
            model = get_mobilenet(width_mult=config.model_width, head_type=config.head_type, se_dims=config.se_dims,
                                  num_classes=10)
    elif teacher_model == "passt":
        model = get_passt(arch="passt_s_swa_p16_128_ap476", pretrained=True, n_classes=10, in_channels=1, fstride=10,
              tstride=10,
              input_fdim=128, input_tdim=998, u_patchout=0, s_patchout_t=0, s_patchout_f=0,
              )
    else:
        model = None

    return model

def train(config):

    # seed = sacred root-seed, which is used to automatically seed random, numpy and pytorch
    seed_sequence = np.random.SeedSequence(1)

    # seed torch, numpy, random with different seeds in main thread
    to_seed = spawn_get(seed_sequence, 2, dtype=int)
    torch.random.manual_seed(to_seed)

    np_seed = spawn_get(seed_sequence, 2, dtype=np.ndarray)
    np.random.seed(np_seed)

    py_seed = spawn_get(seed_sequence, 2, dtype=int)
    random2.seed(py_seed)

    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="DCASE23",
        notes="Pipeline for DCASE23 tasks 1",
        tags=["DCASE23"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name,
        log_model="all"
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath="trained_models",
                                          filename=config.experiment_name + '_{epoch}-{val_loss:.2f}')

    # train dataloader
    train_dl = DataLoader(dataset=get_training_set(config.cache_path, config.resample_rate, config.roll, config.data_dir),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)
    global_train_dl = train_dl

    # test loader
    val_dl = DataLoader(dataset=get_val_set(config.cache_path, config.resample_rate, config.data_dir),
                        worker_init_fn=worker_init_fn,
                        num_workers=config.num_workers,
                        batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = SimpleDCASELitModule(config)
    # create monitor to keep track of learning rate - we want to check the behaviour of our learning rate schedule
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='gpu',
                         devices=1,
                         callbacks=[lr_monitor, checkpoint_callback],
                         default_root_dir="checkpoints/")
    # Start Energy Tracking
    tracker = EmissionsTracker()
    tracker.start()
    # start training and validation
    trainer.fit(pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    # End Energy Tracking
    tracker.stop()

    # so far, save only latest model, not best
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #modelname = "trained_models/model_" + config.architecture + "_" + config.teacher_model + config.experiment_name + "_" + timestr + ".pt"
    modelname = "trained_models/model_" + config.architecture + "_" + config.experiment_name + ".pt"
    torch.save(pl_module.model.state_dict(), modelname)

    # reference can be retrieved in artifacts panel
    # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
    #checkpoint_reference = "christianwi/dcase23/MODEL-RUN_ID:best"

    # download checkpoint locally (if not already cached)
    #run = wandb_logger.init(project="DCASE23")
    #artifact = run.use_artifact(checkpoint_reference, type="model")
    #artifact_dir = artifact.download()

    # load checkpoint
    #model = pl_module.load_from_checkpoint(Path(artifact_dir) / "bestmodel.ckpt")

    #torch.save(pl_module.model.state_dict(), "trained_models/latestmodel.pt")

    # Store total energy
    used_kwh = tracker._total_energy.kWh
    print('Used kwh for training and validation: ', used_kwh)

    trainer.validate(pl_module, ckpt_path="best", dataloaders=val_dl)

    test_dl = DataLoader(dataset=get_test_set(config.cache_path, config.resample_rate, config.data_dir),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    trainer.test(pl_module, ckpt_path="best", dataloaders=test_dl)

    # if quantization=1 is set, we run testing of the quantized model
    #if config.quant:
    #    quant_res = trainer.test(test_dataloaders=test_dl)[0]
    #    quant_res = {"quant_loss": quant_res['loss'], "quant_acc": quant_res['acc']}
    #else:
    #    quant_res = "Run with quantization=1 to obtain quantized results."

    # x = self.forward(x) does not work somehow, error:
    # RuntimeError: Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (512, 512) at dimension 2 of input [1, 51200, 127]
    # solution: x = self.model(x)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parser.')
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="DCASE23")
    parser.add_argument('--num_workers', type=int, default=12)  # number of workers for dataloaders
    parser.add_argument('--architecture', type=str, default="mobilenet")
    parser.add_argument('--modelpath', type=str, default="trained_models")
    parser.add_argument('--cuda', type=int, default=1)

    # dataset
    # location to store resample waveform
    parser.add_argument('--cache_path', type=str, default="datasets/example_data/cached")
    parser.add_argument('--data_dir', type=str, default="datasets/dataset") # location of the dataset
    parser.add_argument('--roll', default=False, action='store_true')  # rolling waveform over time

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    # spectrograms have 1 input channel (RGB images would have 3)
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network
    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--channels_multiplier', type=int, default=2)

    # training
    parser.add_argument('--pretrained_name', type=str, default=None)
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--mixstyle_p', type=float, default=0.0)
    parser.add_argument('--mixstyle_alpha', type=float, default=0.4)
    parser.add_argument('--no_roll', dest='roll', action='store_false')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # learning rate + schedule
    # phases:
    #  1. exponentially increasing warmup phase (for 'warm_up_len' epochs)
    #  2. constant lr phase using value specified in 'lr' (for 'ramp_down_start' - 'warm_up_len' epochs)
    #  3. linearly decreasing to value 'las_lr_value' * 'lr' (for 'ramp_down_len' epochs)
    #  4. finetuning phase using a learning rate of 'last_lr_value' * 'lr' (for the rest of epochs up to 'n_epochs')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--classifier_lr', type=float, default=None)
    parser.add_argument('--last_layer_lr', type=float, default=None)
    parser.add_argument('--features_lr', type=float, default=None)
    parser.add_argument('--warm_up_len', type=int, default=6)
    parser.add_argument('--ramp_down_start', type=int, default=20)
    parser.add_argument('--ramp_down_len', type=int, default=10)
    parser.add_argument('--last_lr_value', type=float, default=0.01)  # relative to 'lr'

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)  # in samples (corresponds to 25 ms)
    parser.add_argument('--hop_size', type=int, default=320)  # in samples (corresponds to 10 ms)
    parser.add_argument('--n_fft', type=int, default=1024)  # length (points) of fft, e.g. 1024 point FFT
    parser.add_argument('--n_mels', type=int, default=256)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=0)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram bins
    parser.add_argument('--fmin', type=int, default=0)  # mel bins are created for freqs. between 'fmin' and 'fmax'
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=1)  # data augmentation: vary 'fmin' and 'fmax'
    parser.add_argument('--fmax_aug_range', type=int, default=1000)

    # knowledge distillation
    parser.add_argument('--temperature', type=int, default=3)
    parser.add_argument('--soft_targets_weight', type=int, default=50)
    parser.add_argument('--teacher_model', type=str, default=None)

    # quantization
    parser.add_argument('--quant', type=int, default=1)

    args = parser.parse_args()
    global_train_dl = 0
    train(args)
