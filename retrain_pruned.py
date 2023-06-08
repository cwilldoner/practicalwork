import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
#from codecarbon import EmissionsTracker
from torchmetrics import Accuracy
import torch.nn.utils.prune as prune
from datasets.audiodataset import get_test_set, get_val_set, get_training_set
from models.mel import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import mixup, mixstyle
from helpers.lr_schedule import exp_warmup_linear_down
import nessi

class SimpleDCASELitModule(pl.LightningModule):
    """
    This is a Pytorch Lightening Module.
    It has several convenient abstractions, e.g. we don't have to specify all parts of the
    training loop (optimizer.step(), loss.backward()) ourselves.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.best_loss = 100
        self.config = config  # results from argparse and contains all configurations for our experiment

        # read config and set mixup and mixstyle configurations
        # self.mixup_alpha = False
        #
        self.mixstyle_p = 0.0
        self.mixstyle_alpha = 0.1

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

        # our model to be trained on the log mel spectrograms
        self.model = torch.load('pruned_simple.pth')

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

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

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
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        # apply mixstyle to spectorgrams
        if self.mixstyle_p > 0:
            x = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha)

        if args.mixup_alpha:
            # Apply Mixup, a very common data augmentation method
            rn_indices, lam = mixup(bs, args.mixup_alpha)  # get shuffled indices and mixing coefficients
            # send mixing coefficients to correct device and make them 4-dimensional
            lam = lam.to(x.device).reshape(bs, 1, 1, 1)
            # mix two spectrograms from the batch
            x = x * lam + x[rn_indices] * (1. - lam)
            # generate predictions for mixed log mel spectrograms
            y_hat = self.model(x)
            # mix the prediction targets using the same mixing coefficients
            samples_loss = (
                    F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                    F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(bs))
            )

        else:
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
        results = {"loss": label_loss, "n_correct_pred": n_correct_pred, "n_pred": len(y),
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

        logs['label_loss'] = avg_label_loss

        self.log_dict(logs)


    def validation_step(self, val_batch, batch_idx):
        # similar to 'training_step' but without any data augmentation
        # pytorch lightening takes care of 'with torch.no_grad()' and 'model.eval()'
        x, y = val_batch
        x = self.mel_forward(x)
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # log validation metric to weights and biases
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save(self.model.state_dict(), 'trained_models/best_model_pruned_statedict.pth')
            torch.save(self.model, 'trained_models/best_model_pruned.pth')
        self.log_dict({'val_loss': avg_loss})

    def test_step(self, test_batch, batch_idx):
        # the test step is used to evaluate the quantized model
        x, y = test_batch
        # x is of shape: batch_size, 1, 320000
        #y = repeat(y, 'b -> (b 10)')
        #x = rearrange(x, 'b c (slices t) -> (b slices) c t', slices=10)
        # quantized model runs only on cpu
        #x, y = x.cpu(), y.cpu()
        #x = self.mel_forward(x)
        y_hat = self.forward(x)
        samples_loss = F.cross_entropy(y_hat, y, reduction="none")
        loss = samples_loss.mean()

        _, preds = torch.max(y_hat, dim=1)
        #n_correct_pred_per_sample = (preds == y)
        #n_correct_pred = n_correct_pred_per_sample.sum()
        #results = {"loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}
        _, preds = torch.max(y_hat, dim=1)
        results = {"preds": preds, "labels": y, "test_loss": loss}
        return results

    def test_epoch_end(self, outputs):
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



def train(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

    # test loader
    val_dl = DataLoader(dataset=get_val_set(config.cache_path, config.resample_rate, config.data_dir),
                        worker_init_fn=worker_init_fn,
                        num_workers=config.num_workers,
                        batch_size=config.batch_size)

    pl_module = SimpleDCASELitModule(config)

    pl_module.model = torch.load(config.modelpath)

    # waveform
    waveform = torch.zeros((1, args.resample_rate * 1))  # 1 seconds waveform
    spectrogram = pl_module.mel(waveform)
    # squeeze in channel dimension
    spectrogram = spectrogram.unsqueeze(1)

    nessi.get_model_size(pl_module.model, 'torch', input_size=spectrogram.size())

    #for name, module in pl_module.model.named_modules():
        # prune 20% of connections in all 2D-conv layers
    #    if isinstance(module, torch.nn.Conv2d):
    #        prune.remove(module, "weight")
        # prune 40% of connections in all linear layers
    #    elif isinstance(module, torch.nn.Linear):
    #        prune.remove(module, "weight")

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
    # start training and validation
    trainer.fit(pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    trainer.validate(pl_module, ckpt_path="best", dataloaders=val_dl)

    test_dl = DataLoader(dataset=get_test_set(config.cache_path, config.resample_rate, config.data_dir),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    trainer.test(pl_module, ckpt_path="best", dataloaders=test_dl)


def validate(trainer, pl_module, config):
    # test loader
    eval_dl = DataLoader(dataset=get_test_set(config.cache_path, config.resample_rate, config.data_dir),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    preds = trainer.test(pl_module, dataloaders=eval_dl)


if __name__ == '__main__':
    torch.set_float32_matmul_precision("medium")
    # simplest form of specifying hyperparameters using argparse
    # IMPORTANT: log hyperparameters to be able to reproduce you experiments!
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="DCASE23")
    parser.add_argument('--num_workers', type=int, default=12)  # number of workers for dataloaders
    parser.add_argument('--modelpath', type=str, default="trained_models")

    # dataset
    # location to store resample waveform
    parser.add_argument('--cache_path', type=str, default="datasets/example_data/cached")
    parser.add_argument('--roll', default=False, action='store_true')  # rolling waveform over time

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    # spectrograms have 1 input channel (RGB images would have 3)
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channels_multiplier', type=int, default=2)

    # training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    # learning rate + schedule
    # phases:
    #  1. exponentially increasing warmup phase (for 'warm_up_len' epochs)
    #  2. constant lr phase using value specified in 'lr' (for 'ramp_down_start' - 'warm_up_len' epochs)
    #  3. linearly decreasing to value 'las_lr_value' * 'lr' (for 'ramp_down_len' epochs)
    #  4. finetuning phase using a learning rate of 'last_lr_value' * 'lr' (for the rest of epochs up to 'n_epochs')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warm_up_len', type=int, default=6)
    parser.add_argument('--ramp_down_start', type=int, default=20)
    parser.add_argument('--ramp_down_len', type=int, default=10)
    parser.add_argument('--last_lr_value', type=float, default=0.001)  # relative to 'lr'

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=2048)  # in samples (corresponds to 25 ms)
    parser.add_argument('--hop_size', type=int, default=744)  # in samples (corresponds to 10 ms)
    parser.add_argument('--n_fft', type=int, default=2048)  # length (points) of fft, e.g. 1024 point FFT
    parser.add_argument('--n_mels', type=int, default=256)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=0)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram bins
    parser.add_argument('--fmin', type=int, default=0)  # mel bins are created for freqs. between 'fmin' and 'fmax'
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=1)  # data augmentation: vary 'fmin' and 'fmax'
    parser.add_argument('--fmax_aug_range', type=int, default=1000)
    parser.add_argument('--data_dir', type=str, default="datasets/dataset")
    args = parser.parse_args()
    # Start Energy Tracking
    # tracker = EmissionsTracker()
    # tracker.start()
    # Start Training
    train(args)
    # End Energy Tracking
    # tracker.stop()

    # Store total energy
    # used_kwh = tracker._total_energy.kWh
    # print('Used kwh for validation: ', used_kwh)
