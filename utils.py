import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import *
import numpy as np
from multiprocessing import cpu_count

def split_sequence(source,
                   target,
                   source_len,
                   target_len):
    """
    Split sequence with sliding window into
    sequences of context features and target.

    Args:
        source (np.array): Source sequence
        target (np.array): Target sequence
        source_len (int): Length of input sequence.
        target_len (int): Length of target sequence.

    Return:
        X (np.array): sequence of features
        y (np.array): sequence of targets
    """
    assert len(source) == len(target), \
            'Source sequence and target sequence should have the same length.'

    X, y = list(), list()
    for i in range(len(source)):
        # Find the end of this pattern:
        src_end = i + source_len
        tgt_end = src_end + target_len
        # Check if beyond the length of sequence:
        if tgt_end > len(source):
            break
        X.append(source[i:src_end, :])
        y.append(target[src_end:tgt_end, :])
    return np.array(X), np.array(y)

def get_datasets_loaders(source,
                         target,
                         test_size,
                         source_len,
                         target_len,
                         batch_size):
    """
    Split the time series into train and test set,
    then create datasets and dataloaders for both datasets.

    Args:
        source (np.array): Source time series dataset.
        target (np.array): Target time series dataset.
        test_size (float): Proportion of test dataset.
        source_len (int): Length of input sequence.
        target_len (int): Length of target sequence.
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
                                          source_len, target_len)
    test_ctx, test_tgt = split_sequence(source[-n:], target[-n:],
                                        source_len, target_len)
    trainset = ArrayDataset([train_ctx, train_tgt])
    testset = ArrayDataset([test_ctx, test_tgt])
    # Get the train and test loaders:
    trainloader = DataLoader(trainset, batch_size, True, num_workers=cpu_count())
    testloader = DataLoader(testset, batch_size, False, num_workers=cpu_count())
    return trainset, testset, trainloader, testloader

def test_fn(model, testloader, metrics, device):
    # Toggle evaluation mode:
    model.eval()
    # Compute the metrics:
    loss = 0
    for src, tgt in testloader:
        with torch.no_grad():
            src = src.transpose(1, 0).to(device)
            tgt = tgt.transpose(1, 0).to(device)
            output = model(src)
            loss += F.mse_loss(output, tgt.squeeze()).item()
    return loss / len(testloader.dataset)

def forecast_fn():
    pass

def plot_forecast_fn():
    pass
