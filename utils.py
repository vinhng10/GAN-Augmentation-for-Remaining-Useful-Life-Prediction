import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataset import *
from multiprocessing import cpu_count

def train_val_split(FD, val_size):
    engines = np.random.permutation(FD[0].unique())
    train_engines, val_engines = sorted(engines[:-val_size]), sorted(engines[-val_size:])
    train_FD = pd.concat([FD[FD[0] == i] for i in train_engines])
    val_FD = pd.concat([FD[FD[0] == i] for i in val_engines])
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

def get_all_rul(FD, max_RUL):
    num_engines = FD[0].unique()
    RULs = list()
    for i in num_engines:
        RULs.append(get_rul((FD[0] == i).sum(), max_RUL))
    return np.concatenate(RULs)

def prepare_plot(FD, engine_idx, window_size):
    fd = FD[FD[0] == engine_idx].iloc[:, 2:].values
    X = list()
    for i in range(len(fd)):
        src_end = i + window_size
        if src_end > len(fd):
            break
        X.append(fd[i:src_end, :])
    return torch.from_numpy(np.array(X)).float()

def plot_rul(fd, model, window_size, n_cols=4, figsize=(20, 15)):
    engines = fd[0].unique()
    n_rows = int(np.ceil(len(engines)/n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i in range(len(engines)):
        # Get test engine:
        X_test = prepare_plot(fd.iloc[:, :-1], engines[i], window_size)
        # Predict on test data:
        with torch.no_grad():
            model.eval()
            pred = model(X_test.to(device))
        rul = get_rul(X_test.shape[0], max_rul)
        pred = pred.cpu().numpy() * max_rul
        ax[i//n_cols, i%n_cols].plot(pred, label="Predicted RUL")
        ax[i//n_cols, i%n_cols].plot(rul, label="Real RUL")
        ax[i//n_cols, i%n_cols].set_title(f"Valid Engine {engines[i]}")
        ax[i//n_cols, i%n_cols].legend()

def prepare_data(FD, window_size, max_RUL, batch_size, shuffle=True):
    num_engines = FD[0].unique()
    src, tgt = list(), list()
    for i in num_engines:
        # Query working conditions & sensor readings:
        X = FD[FD[0] == i].iloc[:, 2:-1].values
        y = FD[FD[0] == i].iloc[:, -1:].values
        # Split source & target with sliding window:
        X, y = split_sequence(X, y, window_size, 1, False)
        src.append(X); tgt.append(y)
    src, tgt = np.concatenate(src), np.concatenate(tgt)
    dataset = ArrayDataset((src, tgt))
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=cpu_count())
    return dataset, dataloader

def split_sequence(source,
                   target,
                   source_len,
                   target_len, 
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
    for i in range(len(source)):
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

def validate(model, validloader, device):
    # Toggle evaluation mode:
    model.eval()
    # Compute the metrics:
    loss = 0
    for src, tgt in validloader:
        with torch.no_grad():
            src, tgt = src.to(device), tgt.to(device)
            pred = model(src)
            loss += F.mse_loss(pred.squeeze(), tgt.squeeze()).item()
    return loss * validloader.batch_size / len(validloader.dataset)

def forecast_fn():
    pass

def plot_forecast_fn():
    pass
