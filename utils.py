import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataset import *
from models import *


def train_val_split(FD, val_size):
    """ Randomly split each subdataset into training & validation set.
        Each subdataset is splitted separately.

        Args:
            FD (list of pd.DataFrame): List of subdatasets.
            val_size (float): Percentage of validation data.

        Return:
            train_FD (np.array): sequence of features
            val_FD (np.array): sequence of targets
    """
    train_FD, val_FD = list(), list()
    for fd in FD:
        n_val = int(len(fd[0].unique()) * val_size)
        engines = np.random.permutation(fd[0].unique())
        train_engines, val_engines = sorted(engines[:-n_val]), sorted(engines[-n_val:])
        train_FD.append(pd.concat([fd[fd[0] == i] for i in train_engines]))
        val_FD.append(pd.concat([fd[fd[0] == i] for i in val_engines]))
    return train_FD, val_FD

def get_rul(cycles, max_RUL):
    """ Get the piece-wise linear degradation process
        corresponding to a multivariate sensor reading.
        Degradation process has 2 phases:
            - 1st phase is constant RUL because the equipment
              only starts to degrade after a period of time.
            - 2nd phase is linearly degrading RUL.

        Args:
            cycles (int): The whole working life cycles.
            max_RUL (int): The maximum RUL for the process
                    (constant in the 1st phase).

        Return:
            RUL (np.array): Remaining useful life.
    """
    constant_RUL = np.full(max(0, cycles-max_RUL), max_RUL)
    linear_RUL = np.arange(min(cycles-1, max_RUL-1), -1, -1)
    return np.concatenate((constant_RUL, linear_RUL)).reshape(-1, 1)

def get_all_rul(fd, max_rul):
    """ Compute the piece-wise linear RUL values
        for all engines in the dataframe.

        Args:
            fd (pd.DataFrame): DataFrame of sensor data of all engines.
            max_rul (int): Predetermined maximum RUL.

        Return:
            ruls (list of float): List of RUL for all engines.
    """
    num_engines = fd[0].unique()
    ruls = list()
    for i in num_engines:
        ruls.append(get_rul((fd[0] == i).sum(), max_rul))
    ruls = np.concatenate(ruls) / max_rul
    return ruls

def prepare_plot(fd, engine_idx, window_size):
    """ Prepare data for visualization.
        Split the selected engine data with sliding window
        into batch of data to feed into RUL prediction model.

        Args:
            fd (pd.DataFrame): DataFrame of sensor data of all engines.
            engine_idx (int): Index of engine in the subdataset.
            window_size (int): Window size.

        Return:
            X (torch.Tensor): Batch of inputs.
    """
    fd = FD[FD[0] == engine_idx].iloc[:, 2:].values
    X = list()
    for i in range(len(fd)):
        src_end = i + window_size
        if src_end > len(fd):
            break
        X.append(fd[i:src_end, :])
    X = torch.from_numpy(np.array(X)).float()
    return X

def plot_rul(model, device, fd, max_rul, window_size, n_cols=4, figsize=(20, 15)):
    """ Plot the predicted RUL against the true RUL of each engine.

        Args:
            model (torch.Module): Trained RUL prediction model.
            device (device): Device to perform visualization.
            fd (DataFrame): DataFrame of sensor data of all engines.
            max_rul (int): Predefined maximum RUL.
            window_size (int): Window size.
            n_cols (int): Number of columns to arrange subplots.
            figsize (tuple): Size of the figure.

        Return:
            None
    """

    engines = fd[0].unique()
    n_rows = int(np.ceil(len(engines)/n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i in range(len(engines)):
        # Get test engine:
        X = prepare_plot(fd.iloc[:, :-1], engines[i], window_size)
        # Predict on test data:
        with torch.no_grad():
            model.to(device)
            model.eval()
            pred = model(X.to(device))
        rul = get_rul(X.shape[0], max_rul)
        pred = pred.squeeze().cpu().numpy() * max_rul
        ax[i//n_cols, i%n_cols].plot(pred, label="Predicted RUL")
        ax[i//n_cols, i%n_cols].plot(rul, label="Real RUL")
        ax[i//n_cols, i%n_cols].set_title(f"Valid Engine {engines[i]}")
        ax[i//n_cols, i%n_cols].legend()

def prepare_data(FD, window_size, step, max_rul, batch_size, trim=True, shuffle=True, pin_memory=True):
    """ Prepare dataset & dataloader for training/testing.

        Args:
            FD (pd.DataFrame): DataFrame of sensor data of all engines.
            window_size (int): Window size.
            step (int): Step for sliding window.
            max_rul (int): Predetermined maximum RUL.
            batch_size (int): Batch size.
            trim (bool): If True, then trim the fisrt part of data
                         to balance the training data.
            shuffle (bool): If True, then shuffle the data in each query.
            pin_memory (bool): If True, then pin memory.

        Return:
            dataset (Dataset): Dataset.
            dataloader (DataLoader): DataLoader for training.
    """

    src, tgt = list(), list()
    for fd in FD:
        for i in fd[0].unique():
            # Query sensor readings:
            X = fd[fd[0] == i].iloc[:, 2:-1].values
            y = fd[fd[0] == i].iloc[:, -1:].values
            # Trim uninteresting part of data:
            t = int(max_rul + 1.5*window_size)
            if trim and len(X) > t:
                X, y = X[-t:], y[-t:]
            # Split source & target with sliding window:
            X, y = split_sequence(X, y, window_size, 1, step, False)
            src.append(X); tgt.append(y)
    src, tgt = np.concatenate(src), np.concatenate(tgt)
    dataset = ArrayDataset((src, tgt))
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=cpu_count(), pin_memory=pin_memory)
    return dataset, dataloader

def split_sequence(source,
                   target,
                   source_len,
                   target_len,
                   step,
                   target_start_next):
    """ Split sequence with sliding window into
        sequences of context features and target.

        Args:
            source (np.array): Source sequence
            target (np.array): Target sequence
            source_len (int): Length of input sequence.
            target_len (int): Length of target sequence.
            target_start_next (bool): If True, target sequence
                    starts on the next time step of last step of source
                    sequence. If False, target sequence starts at the
                    same time step of source sequence.

        Return:
            X (np.array): sequence of features
            y (np.array): sequence of targets
    """
    assert len(source) == len(target), \
            'Source sequence and target sequence should have the same length.'

    X, y = list(), list()
    if not target_start_next:
        target = np.vstack((np.zeros(target.shape[1], dtype=target.dtype), target))
    for i in range(0, len(source), step):
        # Find the end of this pattern:
        src_end = i + source_len
        tgt_end = src_end + target_len
        # Check if beyond the length of sequence:
        if tgt_end > len(target):
            break
        # Split sequences:
        X.append(source[i:src_end, :])
        y.append(target[src_end:tgt_end, :])
    return np.array(X), np.array(y)

def get_datasets_loaders(source,
                         target,
                         test_size,
                         source_len,
                         target_len,
                         target_start_next,
                         batch_size):
    """ Split the time series into train and test set,
        then create datasets and dataloaders for both datasets.

        Args:
            source (np.array): Source time series dataset.
            target (np.array): Target time series dataset.
            test_size (float): Proportion of test dataset.
            source_len (int): Length of input sequence.
            target_len (int): Length of target sequence.
            target_start_next (bool): If True, target sequence
                    starts on the next time step of last step of source
                    sequence. If False, target sequence starts at the
                    same time step of source sequence.
            batch_size (int): Batch size.

        Return:
            trainset (ArrayDataset): Train dataset.
            testset (ArrayDataset): Test dataset.
            trainloader (DataLoader): Train loader.
        testloader (DataLoader): Test loader.
    """
    # Get the split point:
    n = int(len(source) * test_size)
    # Get the train and test datasets:
    train_ctx, train_tgt = split_sequence(source[:-n], target[:-n],
                                          source_len, target_len,
                                          target_start_next)
    test_ctx, test_tgt = split_sequence(source[-n:], target[-n:],
                                        source_len, target_len,
                                        target_start_next)
    trainset = ArrayDataset([train_ctx, train_tgt])
    testset = ArrayDataset([test_ctx, test_tgt])
    # Get the train and test loaders:
    trainloader = DataLoader(trainset, batch_size, True, num_workers=cpu_count())
    testloader = DataLoader(testset, batch_size, False, num_workers=cpu_count())
    return trainset, testset, trainloader, testloader

def prepare_test(test_fd, rul_fd, window_size, max_rul):
    """ Prepare data & target for testing.
        Only take the last datapoints with the size equal
        to the window size for each engine, and use that
        data for testing.

        Args:
            test_fd (pd.DataFrame): DataFrame of sensor data of all engines.
            rul_fd (int): Index of engine in the subdataset.
            window_size (int): Window size.
            max_rul (int): Predefined maximum RUL.

        Return:
            data (torch.Tensor): Batch of inputs.
            rul (torch.Tensor): Batch of targets.
    """
    engines = test_fd[0].unique()
    tmp = []
    for i in engines:
        engine = test_fd[test_fd[0] == i].iloc[:, 2:].values
        if len(engine) < window_size:
            prepend = np.repeat(engine[0:1], max(window_size-len(engine), 0), axis=0)
            tmp.append(np.concatenate((prepend, engine)))
        else:
            tmp.append(engine[-window_size:])
    data = torch.from_numpy(np.array(tmp)).float()
    rul = torch.from_numpy(rul_fd.values / max_rul).float()
    return data, rul

def test(model, test_fd, rul_fd, window_size, max_rul, device):
    """ Test performance of trained model on test datasets.
        This function compute RMSE of the model on all 4 test datasets.

        Args:
            test_fd (pd.DataFrame): DataFrame of sensor data of all engines.
            rul_fd (int): Index of engine in the subdataset.
            window_size (int): Window size.
            max_rul (int): Predefined maximum RUL.

        Return:
            rmse (list of float): List of RMSE on 4 test datasets.
    """
    model.to(device)
    rmse = []
    for i in range(4):
        data, rul = prepare_test(test_fd[i], rul_fd[i], window_size, max_rul)
        with torch.no_grad():
            data, rul = data.to(device), rul.to(device)
            pred = model(data)
            rmse.append(np.sqrt(F.mse_loss(pred.squeeze(), rul.squeeze()).item())*max_rul)
    return rmse
