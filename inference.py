import argparse
import torch
from ex_dcase import SimpleDCASELitModule
import nessi
from models.mel import AugmentMelSTFT
from helpers.utils import print_size_of_model
# import necessary libraries
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch_pruning as tp
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
from torchmetrics import Accuracy
from datasets.evaldataset import get_evaluation_set
from helpers.init import worker_init_fn
import pandas as pd
import os
import librosa as liro
from torchvision import datasets
from torchvision.transforms import ToTensor
from datasets.audiodataset import get_test_set, get_val_set, get_training_set
from helpers.lr_schedule import exp_warmup_linear_down

mnist = False

def create_dataset_config(dataset_path):
    """
    build config dictionary for file paths
    :param dataset_path:
    :return: dictionary with file paths
    """
    dataset_config = {
        "dataset_name": "dcase",  # used to create the cached files path
        # the following files contain metadata about the audio clips, the labels and how to split
        # the files into train and test partitions
        "path": dataset_path,
        "meta_csv": os.path.join(dataset_path, "meta.csv"),
        "train_files_csv": os.path.join(dataset_path, "train.csv"),
        "test_files_csv": os.path.join(dataset_path, "test.csv"),
        "val_files_csv": os.path.join(dataset_path, "evaluate.csv"),
        "evaluation_files_csv": os.path.join(dataset_path, "evaluation.csv")

    }
    return dataset_config


def do_inference(config):
    """
    Running Inference on an audio clip.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model
    pl_module = SimpleDCASELitModule(config)

    #mymodel = torch.load(modelpath)
    #pl_module.model.load_state_dict(torch.load(modelpath), strict=False)
    #print(pl_module.model.state_dict())

    pl_module.model = torch.load(config.modelpath)

    # disable randomness, dropout, etc...
    pl_module.model.eval()
    pl_module.model.to(device)

    print("Model parameters:")
    print(sum(p.numel() for p in pl_module.model.parameters()))

    # MNIST dataset
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    train_subset, val_subset = torch.utils.data.random_split(
        train_data, [50000, 10000], generator=torch.Generator().manual_seed(1))

    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )

    if mnist:
        trainset = train_subset
        valset = val_subset
        testset = test_data
    else:
        trainset = get_training_set(config.cache_path, config.resample_rate, config.roll, "../malach23/malach/datasets/dataset")
        valset = get_val_set(config.cache_path, config.resample_rate, "../malach23/malach/datasets/dataset")
        testset = get_test_set(config.cache_path, config.resample_rate, "../malach23/malach/datasets/dataset")

    from torch.utils.data import DataLoader

    # train dataloader
    train_dl = DataLoader(dataset=trainset,
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    # validation loader
    val_dl = DataLoader(dataset=valset,
                        worker_init_fn=worker_init_fn,
                        num_workers=config.num_workers,
                        batch_size=config.batch_size)

    # test loader
    test_dl = DataLoader(dataset=testset,
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    if mnist is False:
        # model to preprocess waveform into mel spectrograms
        mel = AugmentMelSTFT(n_mels=config.n_mels, sr=config.resample_rate, win_length=config.window_size, hopsize=config.hop_size, n_fft=config.n_fft)
        mel.to(device)
        mel.eval()

        # waveform
        dataset_config = create_dataset_config(config.data_dir)
        test_files = pd.read_csv(dataset_config['evaluation_files_csv'], sep='\t')['filename'].values.reshape(-1)
        directory = os.getcwd()
        filepath = os.path.join(config.data_dir, test_files[2])

        waveform, _ = liro.load(filepath, sr=config.resample_rate)
        waveform = torch.from_numpy(waveform).to(device)
        waveform = waveform[None, :]
        print(waveform.shape)
        waveform = torch.zeros((1, config.resample_rate * 1)).to(device)  # 1 seconds waveform
        print(waveform.shape)
        spectrogram = mel(waveform).to(device)
        # squeeze in channel dimension
        spectrogram = spectrogram.unsqueeze(1)
        print(spectrogram.shape)
        y_hat = pl_module.model.forward(spectrogram)
        print(torch.max(y_hat, dim=1))
        #nessi.get_model_size(pl_module.model, 'torch', input_size=spectrogram.size())

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dl:
                #images = images[None, :]
                images = images.to(device)
                print(images.shape)
                test_output = pl_module.model.forward(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (pred_y == labels.to(device)).sum().item() / float(labels.to(device).size(0))
                pass

        print('Test Accuracy of the model on the asc test images: %.2f' % accuracy)

        sample = next(iter(test_dl))
        imgs, lbls = sample
        imgs, lbls = imgs.to(device), lbls.to(device)
        actual_number = lbls[:10].cpu().numpy()
        test_output = pl_module.model(imgs[:10])
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
        print(f'Prediction classes: {pred_y}')
        print(f'Actual classes: {actual_number}')

    else:
        test_data = datasets.MNIST(
            root='data',
            train=False,
            transform=ToTensor()
        )
        # test loader
        test_dl = DataLoader(dataset=test_data,
                             worker_init_fn=worker_init_fn,
                             num_workers=config.num_workers,
                             batch_size=config.batch_size)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dl:
                print(images.shape)
                test_output = pl_module.model(images.to(device))
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (pred_y == labels.to(device)).sum().item() / float(labels.to(device).size(0))
                pass

        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)

        sample = next(iter(test_dl))
        imgs, lbls = sample
        imgs, lbls = imgs.to(device), lbls.to(device)
        actual_number = lbls[:10].cpu().numpy()
        test_output = pl_module.model(imgs[:10])
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
        print(f'Prediction number: {pred_y}')
        print(f'Actual number: {actual_number}')

    if config.prune:
        print("==============================================================================")
        print("Prune model")
        # save unpruned model
        torch.save(pl_module.model, "trained_models/unpruned.pth")

        print("Unpruned size:")
        print("Model parameters:")
        print(sum(p.numel() for p in pl_module.model.parameters()))
        print_size_of_model(pl_module.model)

        if mnist is True:
            example_input = imgs[0]
            example_input = example_input[None, :]
        else:
            sample = next(iter(test_dl))
            imgs, lbls = sample
            imgs, lbls = imgs.to(device), lbls.to(device)
            example_input = imgs[0]

        prune_highlevel(device, config, pl_module.model, example_input, imgs, lbls, train_dl, val_dl, test_dl)


def prune_highlevel(device, config, model, example_inputs, imgs, lbls, train_dl, val_dl, test_dl):
    num_epochs = 10

    # 0. importance criterion for parameter selections
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

    # 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and m.out_channels == 10:
            ignored_layers.append(m)  # DO NOT prune the final classifier!

    # 2. Pruner initialization
    iterative_steps = 5  # You can prune your model to the target sparsity iteratively.
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        global_pruning=False,  # If False, a uniform sparsity will be assigned to different layers.
        importance=imp,  # importance criterion for parameter selection
        iterative_steps=iterative_steps,  # the number of iterations to achieve target sparsity
        ch_sparsity=0.2,  # remove 20% channels
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        # 3. the pruner.step will remove some channels from the model with least importance
        pruner.step()

        # 4. Do whatever you like here, such as finetuning
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        #print(model)
        print(model(example_inputs).shape)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )
        # finetune your model here
        # finetune(model)
        # ...
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        schedule_lambda = \
            exp_warmup_linear_down(config.warm_up_len, config.ramp_down_len, config.ramp_down_start,
                                   config.last_lr_value)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        # Train the model
        total_step = len(train_dl)

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_dl):
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                #print(labels.shape)
                #print(output.shape)
                loss = F.cross_entropy(output, labels, reduction="none")
                #print(loss)
                # clear gradients for this training step
                optimizer.zero_grad()
                # backpropagation, compute gradients
                loss.mean().backward()
                # apply gradients
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                             loss.mean()))

    print("Pruned model parameters:")
    print(sum(p.numel() for p in model.parameters()))

    if mnist is True:
        actual_number = lbls[:10].cpu().numpy()
        test_output = model(imgs[:10])
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
        print(f'Prediction number: {pred_y}')
        print(f'Actual number: {actual_number}')

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dl:
                test_output = model(images.to(device))
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (pred_y == labels.to(device)).sum().item() / float(labels.to(device).size(0))
                pass

        print('Test Accuracy of the pruned model on the 10000 test images: %.2f' % accuracy)
    else:
        sample = next(iter(test_dl))
        imgs, lbls = sample
        imgs, lbls = imgs.to(device), lbls.to(device)


        actual_number = lbls[:10].cpu().numpy()
        test_output = model(imgs[:10])
        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
        print(f'Prediction number: {pred_y}')
        print(f'Actual number: {actual_number}')


    torch.save(model, 'trained_models/pruned.pth')


def finetune(device, config, model, train_dl, num_epochs):

    model.train()
    import torch.nn.functional as F
    from helpers.lr_schedule import exp_warmup_linear_down
    import torch.nn as nn
    from torch import optim
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    schedule_lambda = \
        exp_warmup_linear_down(config.warm_up_len, config.ramp_down_len, config.ramp_down_start,
                               config.last_lr_value)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
    # Train the model
    total_step = len(train_dl)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dl):
            # gives batch data, normalize x when iterate train_loader
            #b_x = Variable(images)  # batch x
            #b_y = Variable(labels)  # batch y

            labels = labels.to(device)
            #b_x = b_x.to(device)
            #b_y = b_y.to(device)
            #print(images.shape)
            #print(b_x.shape)
            #print(b_y.shape)
            #print(labels.shape)
            output = model(images.to(device))
            #output = model(b_x)
            #print(output[:,0])

            #preds = torch.max(output, 1)[1].data.cpu().numpy().squeeze()
            #preds = torch.tensor(preds).float()
            #preds = torch.tensor(preds, requires_grad=True).to(device)
            print("y_hat: ", output.shape)
            print("y: ", labels.shape)
            #loss_fn = nn.CrossEntropyLoss()
            #loss = loss_fn(preds.float(), labels.float())
            loss = F.cross_entropy(output, labels, reduction="none")

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    return model

def retrain(config, model):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="PracticalWork",
        notes="Pipeline for Practical Work",
        tags=["PracticalWork"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name,
        log_model="all"
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath="trained_models",
                                          filename=config.experiment_name + '_{epoch}-{val_loss:.2f}')


    # MNIST dataset
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    train_subset, val_subset = torch.utils.data.random_split(
        train_data, [50000, 10000], generator=torch.Generator().manual_seed(1))

    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )
    if mnist is True:
        trainset = train_subset
        valset = val_subset
        testset = test_data
    else:
        trainset = get_training_set(config.cache_path, config.resample_rate, config.roll, config.data_dir)
        valset = get_val_set(config.cache_path, config.resample_rate, config.data_dir)
        testset = get_test_set(config.cache_path, config.resample_rate, config.data_dir)


    # train dataloader
    train_dl = DataLoader(dataset=trainset,
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    # validation loader
    val_dl = DataLoader(dataset=valset,
                        worker_init_fn=worker_init_fn,
                        num_workers=config.num_workers,
                        batch_size=config.batch_size)

    # test loader
    test_dl = DataLoader(dataset=testset,
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

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
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    trainer.test(model, ckpt_path="best", dataloaders=test_dl)
def prune_advanced(config, device, input):
    # model is loaded, start pruning

    # load model
    pl_module_prune = SimpleDCASELitModule(config)

    pl_module_prune.model = torch.load('trained_models/unpruned.pth')

    # disable randomness, dropout, etc...
    pl_module_prune.model.eval()
    pl_module_prune.model.to(device)

    print("tp.DependencyGraph().build_dependency")
    # 1. build dependency graph for resnet18
    DG = tp.DependencyGraph().build_dependency(pl_module_prune.model, example_inputs=input)

    # 2. Specify the to-be-pruned channels. Here we prune those channels indexed by [2, 6, 9].
    group = DG.get_pruning_group(pl_module_prune.model.in_c, tp.prune_conv_in_channels, idxs=[2, 6, 9])

    # 3. prune all grouped layers that are coupled with model.conv1 (included).
    if DG.check_pruning_group(group):  # avoid full pruning, i.e., channels=0.
        group.prune()

    # 4. Save & Load
    pl_module_prune.model.zero_grad()  # We don't want to store gradient information
    torch.save(pl_module_prune.model, 'trained_models/pruned.pth')  # without .state_dict
    model = torch.load('trained_models/pruned.pth')  # load the model object

    print("Pruned model parameters:")
    print(sum(p.numel() for p in pl_module_prune.model.parameters()))
    #nessi.get_model_size(model, 'torch', input_size=spectrogram.size())

if __name__ == '__main__':
    # simplest form of specifying hyperparameters using argparse
    # IMPORTANT: log hyperparameters to be able to reproduce you experiments!
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="DCASE23")
    parser.add_argument('--num_workers', type=int, default=12)  # number of workers for dataloaders
    parser.add_argument('--architecture', type=str, default="mobilenet")
    parser.add_argument('--modelpath', type=str, default="trained_models")
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--channel_width', type=str, default='24, 48, 72')

    # dataset
    # location to store resample waveform
    parser.add_argument('--cache_path', type=str, default="datasets/example_data/cached")
    parser.add_argument('--data_dir', type=str, default="../malach23/malach/datasets/evaluation_dataset") # location of the dataset
    parser.add_argument('--roll', default=False, action='store_true')  # rolling waveform over time

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    # spectrograms have 1 input channel (RGB images would have 3)
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channels_multiplier', type=int, default=2)

    # training
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--pretrained_name', type=str, default=None)
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--batch_size', type=int, default=64)
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

    parser.add_argument('--prune', default=False)

    args = parser.parse_args()

    # get class of audio-clip
    do_inference(args)

