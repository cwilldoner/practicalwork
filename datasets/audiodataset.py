import pandas as pd
import os
from sklearn import preprocessing
from torch.utils.data import Dataset
import torch
import numpy as np
import librosa
from typing import Callable, List


# the directory 'datasets/example_data' contains 200 wav files that we use to test this setup
# dataset_dir = "/media/noah/Samsung_T5/dcase"

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


class BasicAudioDataset(Dataset):
    def __init__(self, dc: dict, sr: int = 32000, cache_path: str = None):
        """
        @param dc: dataset config
        @param sr: sampling rate
        @param cache_path: cache path to store resampled waveforms
        return: waveform, label
        """
        df = pd.read_csv(dc["meta_csv"], sep="\t")
        le = preprocessing.LabelEncoder()  # sklearn label encoder to transform strings into numbers
        self.labels = torch.from_numpy(le.fit_transform(df[['scene_label']].values.reshape(-1))).long()
        self.c_names = list(le.classes_)
        self.files = df[['filename']].values.reshape(-1)
        self.sr = sr
        self.dc = dc
        # why do we want to cache the audio clips?
        # resampling is time-consuming -> we only do it once and save the resampled signals as a pytorch tensors
        if cache_path is not None:
            self.cache_path = os.path.join(cache_path, dc["dataset_name"] + f"_r{self.sr}", "files_cache")
            os.makedirs(self.cache_path, exist_ok=True)
        else:
            self.cache_path = None

    def __getitem__(self, index):
        if self.cache_path:
            cpath = os.path.join(self.cache_path, str(index) + ".pt")
            try:
                sig = torch.load(cpath)
            except FileNotFoundError:
                # not yet cached, we have to do the resampling
                sig, _ = librosa.load(os.path.join(self.dc["path"], self.files[index]), sr=self.sr, mono=True)
                sig = torch.from_numpy(sig[np.newaxis])
                torch.save(sig, cpath)
        else:
            # no caching used
            sig, _ = librosa.load(os.path.join(self.dc["path"], self.files[index]), sr=self.sr, mono=True)
            sig = torch.from_numpy(sig[np.newaxis])
        return sig, self.labels[index]

    def __len__(self):
        return len(self.files)


class SimpleSelectionDataset(Dataset):
    """A dataset that selects a subset from a dataset based on a set of sample ids.
        Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset: Dataset, available_indices: List[int]):
        """
        @param dataset: dataset to load data from
        @param available_indices: available indices of samples for 'training', 'testing'
        return: x, label
        """
        self.available_indices = available_indices
        self.dataset = dataset

    def __getitem__(self, index):
        x, label = self.dataset[self.available_indices[index]]
        return x, label

    def __len__(self):
        return len(self.available_indices)


class PreprocessDataset(Dataset):
    """A base preprocessing dataset representing a preprocessing step of a Dataset preprocessed on the fly.
    Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset: Dataset, preprocessor: Callable):
        self.dataset = dataset
        if not callable(preprocessor):
            print("preprocessor: ", preprocessor)
            raise ValueError('preprocessor should be callable')
        self.preprocessor = preprocessor

    def __getitem__(self, index):
        return self.preprocessor(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


def get_roll_func(axis=1, shift_range=10000):
    # roll waveform over time
    def roll_func(batch):
        x = batch[0]  # the waveform
        others = batch[1:]  # label + possible other metadata in batch
        sf = int(np.random.random_integers(-shift_range, shift_range))
        return x.roll(sf, axis), *others

    return roll_func


# commands to create the datasets for training and testing
def get_training_set(cache_path="datasets/example_data/cached", resample_rate=32000, roll=False, data_dir="datasets/example_data"):
    dataset_config = create_dataset_config(data_dir)
    # get filenames of clips in training set
    train_files = pd.read_csv(dataset_config['train_files_csv'], sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    # get indices of training clips
    train_indices = list(meta[meta['filename'].isin(train_files)].index)
    # first create dataset and then use 'SimpleSelectionDataset' to select train partition
    ds = SimpleSelectionDataset(
        BasicAudioDataset(dataset_config, sr=resample_rate, cache_path=cache_path),
        train_indices)
    # use the PreprocessDataset to add data augmentations applied to raw waveforms here
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    return ds


def get_val_set(cache_path="example_data/cached", resample_rate=32000, data_dir="datasets/example_data"):
    dataset_config = create_dataset_config(data_dir)
    # get filenames of clips in validation set
    test_files = pd.read_csv(dataset_config['val_files_csv'], sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    # get indices of validation clips
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    # first create dataset and then use 'SimpleSelectionDataset' to select validation partition
    ds = SimpleSelectionDataset(
        BasicAudioDataset(dataset_config, sr=resample_rate, cache_path=cache_path),
        test_indices)
    return ds


def get_test_set(cache_path="dataset/example_data/cached", resample_rate=32000, data_dir="datasets/example_data"):
    """
    currently just interface for import
    :return:
    """
    dataset_config = create_dataset_config(data_dir)
    # get filenames of clips in test set
    test_files = pd.read_csv(dataset_config['test_files_csv'], sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    # get indices of validation clips
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    # first create dataset and then use 'SimpleSelectionDataset' to select evaluation
    ds = SimpleSelectionDataset(
        BasicAudioDataset(dataset_config, sr=resample_rate, cache_path=cache_path),
        test_indices)
    return ds

def get_evaluation_set(cache_path="dataset/example_data/cached", resample_rate=32000, data_dir="datasets/evaluation_dataset"):
    """
    currently just interface for import
    :return:
    """
    dataset_config = create_dataset_config(data_dir)
    # get filenames of clips in test set
    test_files = list(pd.read_csv(dataset_config['evaluation_files_csv'], sep='\t')['filename'].index)
    #test_files = pd.read_csv(dataset_config['evaluation_files_csv'], sep='\t')['filename'].values.reshape(-1)
    # get indices of validation clips
    #test_indices = list(test_files[test_files['filename']].index) #list(meta[meta['filename'].isin(test_files)].index)
    # first create dataset and then use 'SimpleSelectionDataset' to select evaluation
    ds = SimpleSelectionDataset(
        BasicAudioDataset(dataset_config, sr=resample_rate, cache_path=cache_path),
        test_files)
    return ds