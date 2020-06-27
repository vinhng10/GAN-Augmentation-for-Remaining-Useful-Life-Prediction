import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import *
import numpy as np
from multiprocessing import cpu_count

def get_RUL(cycles, max_RUL):
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
    return np.concatenate((constant_RUL, linear_RUL))

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

def test_fn(encoder, decoder, testloader, device):
    # Toggle evaluation mode:
    encoder.eval()
    decoder.eval()
    # Compute the metrics:
    loss = 0
    for src, tgt in testloader:
        with torch.no_grad():
            src, tgt = src.to(device), tgt.to(device)
            _, hidden = encoder(src)
            outputs, _ = decoder(hidden, len(tgt[0]))
            loss += F.mse_loss(outputs, tgt).item()
    return loss / testloader.batch_size

def forecast_fn():
    pass

def plot_forecast_fn():
    pass
