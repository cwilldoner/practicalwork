import argparse
import torch
from ex_dcase import SimpleDCASELitModule
from codecarbon import EmissionsTracker
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
#from codecarbon import EmissionsTracker
from torchmetrics import Accuracy
import torch.nn.utils.prune as prune
from datasets.evaldataset import get_evaluation_set
from helpers.init import worker_init_fn
import pandas as pd
import os
import librosa as liro

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

def audio_tagging(config):
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
    #print(waveform.shape)
    #waveform = torch.zeros((1, config.resample_rate * 1)).to(device)  # 1 seconds waveform
    print(waveform.shape)
    spectrogram = mel(waveform).to(device)
    # squeeze in channel dimension
    spectrogram = spectrogram.unsqueeze(1)

    # Initialize tracker
    tracker = EmissionsTracker()
    # Start tracker
    tracker.start()
    y_hat = pl_module.model.forward(spectrogram)
    print(torch.max(y_hat, dim=1))
    # Stop tracker
    tracker.stop()
    # Store total energy
    used_kwh = tracker._total_energy.kWh
    print('Used kwh for inference: ', used_kwh)

    #test_dl = DataLoader(dataset=get_evaluation_set(config.cache_path, config.resample_rate, config.data_dir),
    #                     worker_init_fn=worker_init_fn,
    #                     num_workers=config.num_workers,
    #                     batch_size=config.batch_size)

    #trainer = pl.Trainer(accelerator='gpu', devices=1).test(pl_module, dataloaders=test_dl)



    #nessi.get_model_size(pl_module.model, 'torch', input_size=spectrogram.size())
    #print(sum(p.numel() for p in pl_module.model.parameters()))

    #print("Are the modules pruned:")
    #for name, module in pl_module.model.named_modules():
    #    print(name, torch.nn.utils.prune.is_pruned(module))
    if config.prune:
        print("Prune model")
        # save unpruned model
        torch.save(pl_module.model, "unpruned.pth")

        #print('Used kwh for inference (input is 1s spectrogram): ', used_kwh)

        #print_size_of_model(pl_module.model)

        print("Unpruned size:")
        #print_size_of_model(pl_module.model)


        prune_simple(config, device, spectrogram, test_dl)


        #prune_advanced(config, device, spectrogram)


def prune_advanced(config, device, spectrogram):
    # model is loaded, start pruning

    # load model
    pl_module_prune = SimpleDCASELitModule(config)

    pl_module_prune.model.load_state_dict(torch.load('unpruned.pth'), strict=False)

    # disable randomness, dropout, etc...
    pl_module_prune.model.eval()
    pl_module_prune.model.to(device)

    print("tp.DependencyGraph().build_dependency")
    # 1. build dependency graph for resnet18
    DG = tp.DependencyGraph().build_dependency(pl_module_prune.model, example_inputs=spectrogram)

    # 2. Specify the to-be-pruned channels. Here we prune those channels indexed by [2, 6, 9].
    group = DG.get_pruning_group(pl_module_prune.model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9])

    # 3. prune all grouped layers that are coupled with model.conv1 (included).
    if DG.check_pruning_group(group):  # avoid full pruning, i.e., channels=0.
        group.prune()

    # 4. Save & Load
    pl_module_prune.model.zero_grad()  # We don't want to store gradient information
    torch.save(pl_module_prune.model, 'pruned.pth')  # without .state_dict
    model = torch.load('pruned.pth')  # load the model object

    nessi.get_model_size(model, 'torch', input_size=spectrogram.size())


def prune_simple(config, device, spectrogram, test_dl):

    # load model
    pl_module_prune = SimpleDCASELitModule(config)

    pl_module_prune.model.load_state_dict(torch.load('unpruned.pth'), strict=False)

    # disable randomness, dropout, etc...
    pl_module_prune.model.eval()
    pl_module_prune.model.to(device)

    for name, module in pl_module_prune.model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=0.4, n=1, dim=0)
            #prune.remove(module, "weight")
        # prune 40% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name='weight', amount=0.2, n=1, dim=0)
            #prune.remove(module, "weight")

    # Model Complexity Report
    nessi.get_model_size(pl_module_prune.model, 'torch', input_size=spectrogram.size())

    torch.save(pl_module_prune.model, 'pruned_simple.pth')
    #torch.save(pl_module_prune.model.state_dict(), 'pruned_simple.pth')

    print("Pruned size:")
    print_size_of_model(pl_module_prune.model)
    print(sum(p.numel() for p in pl_module_prune.model.parameters()))
    trainer = pl.Trainer(accelerator='gpu', devices=1).test(pl_module_prune, dataloaders=test_dl)

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

    # dataset
    # location to store resample waveform
    parser.add_argument('--cache_path', type=str, default="datasets/example_data/cached")
    parser.add_argument('--data_dir', type=str, default="datasets/evaluation_dataset") # location of the dataset
    parser.add_argument('--roll', default=False, action='store_true')  # rolling waveform over time

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    # spectrograms have 1 input channel (RGB images would have 3)
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channels_multiplier', type=int, default=2)

    # training
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

    parser.add_argument('--prune', type=bool, default=False)

    args = parser.parse_args()

    # get class of audio-clip
    audio_tagging(args)

