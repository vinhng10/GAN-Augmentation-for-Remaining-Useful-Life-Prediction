import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from dataset import *
from models import *

##########################################################################
#                        RUL Estimation Utilities                        #
##########################################################################
def train_val_split(FD, val_size):
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
    num_engines = fd[0].unique()
    ruls = list()
    for i in num_engines:
        ruls.append(get_rul((fd[0] == i).sum(), max_rul))
    ruls = np.concatenate(ruls) / max_rul
    return ruls

def prepare_plot(FD, engine_idx, window_size):
    fd = FD[FD[0] == engine_idx].iloc[:, 2:].values
    X = list()
    for i in range(len(fd)):
        src_end = i + window_size
        if src_end > len(fd):
            break
        X.append(fd[i:src_end, :])
    return torch.from_numpy(np.array(X)).float()

def plot_rul(model, device, fd, max_rul, window_size, n_cols=4, figsize=(20, 15)):
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
    model.to(device)
    rmse = []
    for i in range(4):
        data, rul = prepare_test(test_fd[i], rul_fd[i], window_size, max_rul)
        with torch.no_grad():
            data, rul = data.to(device), rul.to(device)
            pred = model(data)
            rmse.append(np.sqrt(F.mse_loss(pred.squeeze(), rul.squeeze()).item())*max_rul)
    return rmse

##########################################################################
#                              GAN Utilities                             #
##########################################################################
def get_all_condition(fd, max_rul):
    num_engines = fd[0].unique()
    rul_list = list()
    for i in num_engines:
        cycles = (fd[0] == i).sum()
        rul_list.append(get_rul(cycles, max_rul))
    ruls = np.concatenate(rul_list) / max_rul
    return ruls

def prepare_gan_data(FD, window_size, step, max_rul, batch_size, trim=True, shuffle=True, pin_memory=True):
    src, tgt = list(), list()
    for fd in FD:
        for i in fd[0].unique():
            # Query sensor readings:
            X = fd[fd[0] == i].iloc[:, 2:-1].values
            y = fd[fd[0] == i].iloc[:, -1:].values
            # Trim uninteresting part of data:
            t = int(max_rul + 1.5*window_size)
            if trim and (len(X) > t):
                X, y = X[-t:], y[-t:]
            # Split source & target with sliding window:
            X, y = split_sequence(X, y, window_size, 1, step, False)
            src.append(X); tgt.append(y)
    src, tgt = np.concatenate(src), np.concatenate(tgt)
    dataset = ArrayDataset((src, tgt))
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=cpu_count(), pin_memory=pin_memory)
    return dataset, dataloader

def add_noise(tensor, mean, sigma):
    return tensor + torch.FloatTensor(np.random.normal(mean, sigma, tensor.shape)).to(tensor.device)

def generator_loss(D, fake, condition, instance_noise=0.01):
    """GAN Generator Loss.

    Args:
        D: Discriminator.
        condition (batch, seq_len, 2): Conditional input.
        fake (batch, seq_len, feature_size): Fake sequence.

    Returns:
        loss: Average cross entropy of fake sequence.
    """
    # Instance noise:
    if instance_noise:
        fake = add_noise(fake, 0, instance_noise)
    # Feed fake sequence to discriminator:
    pred = D(fake, condition)
    # Prepare target (as 1s to confused discriminator):
    target = torch.ones(fake.shape[0], device=fake.device)
    # Binary cross entropy loss:
    loss = F.binary_cross_entropy(pred, target)
    return loss

def discriminator_loss(D, real, fake, condition, label_flip=True, label_smooth=True, instance_noise=0.01):
    """GAN Discriminator Loss.

    Args:
        D: Discriminator.
        condition (batch, seq_len, 2): Conditional input.
        real (batch, seq_len, feature_size): Real sequence.
        fake (batch, seq_len, feature_size): Fake sequence.

    Returns:
        loss_real: Average cross entropy of real sequence.
        avg_pred_real: Average of discriminator predicting real
                sequence as real.
        loss_fake: Average cross entropy of fake sequence.
        avg_pred_fake: Average of discriminator predicting fake
                sequence as fake.
    """
    # Instance noise:
    if instance_noise:
        real = add_noise(real, 0, instance_noise)
        fake = add_noise(fake, 0, instance_noise)
    # Prepare target (1s for real and 0s for fake):
    target_real = torch.ones(real.shape[0], device=real.device)
    target_fake = torch.zeros(fake.shape[0], device=fake.device)
    if label_smooth:
        target_real = torch.FloatTensor(np.random.uniform(0.8, 1.0, real.shape[0])).to(real.device)
        target_fake = torch.FloatTensor(np.random.uniform(0.0, 0.2, fake.shape[0])).to(fake.device)
    if label_flip:
        if torch.rand(1).item() < 0.1:
            target_real, target_fake = target_fake, target_real
    # Feed both real and fake sequence to discriminator:
    pred_real = D(real, condition)
    pred_fake = D(fake, condition)
    # Binary cross entropy loss:
    loss_real = F.binary_cross_entropy(pred_real, target_real)
    loss_fake = F.binary_cross_entropy(pred_fake, target_fake)
    # Compute mean of prediction for tracking:
    avg_pred_real = pred_real.mean()
    avg_pred_fake = pred_fake.mean()
    return loss_real, avg_pred_real, loss_fake, avg_pred_fake

def plot_generated(G, params, device):
    # Generate fake data:
    G = G.to(device)
    G.eval()
    noise = torch.randn((8, params["window_size"], params["noise_size"]), device=device)
    condition = torch.FloatTensor(np.random.uniform(size=(8, 1, 1))).expand(-1, params["window_size"], -1).to(device)
    with torch.no_grad():
        fake = G(noise, condition)
    fake = fake.cpu().numpy()
    condition = condition.cpu().numpy()
    fig, ax = plt.subplots(2, 4, figsize=(20, 6))
    for i in range(8):
        ax[i//4, i%4].plot(fake[i])
        ax[i//4, i%4].set_title("rul %.3f" % (condition[i][0][0]))
        ax[i//4, i%4].axis("off")
    plt.show()
        
def train(G=None, D=None, history=None, trainloader=None, device="cpu", params=None):
    total_steps = math.ceil(len(trainloader.dataset) / params["batch_size"])
    new_line = total_steps // 2

    # Generator & Discriminator:
    if G is None or D is None:
        G = Generator(noise_size=params["noise_size"],
                      hidden_size=params["hidden_size"],
                      output_size=params["feature_size"],
                      num_layers=params["num_layers"],
                      dropout=params["dropout"])
        D = Discriminator(input_size=params["feature_size"],
                          hidden_size=params["hidden_size"],
                          num_layers=params["num_layers"],
                          dropout=params["dropout"],
                          bidirectional=params["bidirectional"])
        G = G.to(device)
        D = D.to(device)

    # History:
    if history is None:
        history = {
            "lossD": [],
            "lossG": [],
            "D_real": [],
            "D_fake": []
        }

    # Optimizer:
    optimG = optim.Adam(G.parameters(), lr=params["G_lr"], betas=(params["momentum"], 0.999))
    optimD = optim.Adam(D.parameters(), lr=params["D_lr"], betas=(params["momentum"], 0.999))

    # Main training loop:
    for epoch in range(params["max_epochs"]):
        for i, (sequence, condition) in enumerate(trainloader):
            G.train(); D.train()
            #################################################################
            # Train Discriminator: maximize log(D(x)) + log(1 - D(G(x)))    #
            #################################################################
            D.zero_grad()
            # Form real and fake batch separately:
            noise = torch.randn((condition.shape[0], params["window_size"], params["noise_size"]), device=device)
            condition = condition.expand(-1, params["window_size"], -1).to(device)
            real = sequence.to(device)
            fake = G(noise, condition)
            # Forward real and fake batch through D:
            lossD_real, avg_pred_real, lossD_fake, avg_pred_fake = discriminator_loss(D, real, fake, condition, 
                                                                                      params["label_flip"], 
                                                                                      params["label_smooth"], 
                                                                                      params["instance_noise"])
            # Compute the total loss for discriminator:
            lossD = lossD_real + lossD_fake
            # Backpropagation:
            lossD.backward()
            # Update D:
            optimD.step()

            ################################################################
            # Train Generator: maximize log(D(G(z)))                       #
            ################################################################
            G.zero_grad()
            # Generate fake batch:
            fake = G(noise, condition)
            # Forward real and fake batch through D:
            lossG = generator_loss(D, fake, condition, params["instance_noise"])
            # Backpropagation:
            lossG.backward()
            # Update D:
            optimG.step()

            #################################################################
            # Print Training Progress                                       #
            #################################################################
            # Record training progress:
            history["lossD"].append(lossD.item())
            history["lossG"].append(lossG.item())
            history["D_real"].append(avg_pred_real.item())
            history["D_fake"].append(avg_pred_fake.item())

            # Print out training progress (on same line):
            stats = '[%3d/%3d][%3d/%3d]\tLossD: %.4f, LossG: %.4f, D_Real: %.4f, D_fake: %.4f' \
                    % (epoch+1, params["max_epochs"], i, total_steps, lossD.item(), lossG.item(), \
                      avg_pred_real.item(), avg_pred_fake.item())
            print('\r' + stats, end="", flush=True)

            # Print out training progress (on different line):
            if (i % new_line == 0):
                print('\r' + stats)
        
        # Visually tracking th quality of generated data:
        plot_generated(G, params, device) 

    # Done:
    return G, D, history



