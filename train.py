import os, sys, math
from datetime import date

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, mIoU, ConfusionMatrix
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

from dataset import *
from models import *
from utils import *

def add_noise(tensor, mean, sigma):
    """ Add Gaussian noise to the tensor.
        The noise has mean and standard deviation
        as defined in the arguements.

    Args:
        tensor (torch.Tensor): Tensor to add noise.
        mean (float): Mean of the noise.
        sigma (float): Standard deviation of the noise.

    Returns:
        tensor (torch.Tensor): Noisy tensor.
    """
    return tensor + torch.FloatTensor(np.random.normal(mean, sigma, tensor.shape)).to(tensor.device)

def generator_loss(D, fake, condition, instance_noise=0.01):
    """ GAN Generator Loss.

    Args:
        D: Discriminator.
        condition (batch, seq_len, 2): Conditional input.
        fake (batch, seq_len, feature_size): Fake sequence.
        instance_noise (float): Standard deviation for noise
            used in instance noise.

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
    """ GAN Discriminator Loss.

    Args:
        D: Discriminator.
        condition (batch, seq_len, 2): Conditional input.
        real (batch, seq_len, feature_size): Real sequence.
        fake (batch, seq_len, feature_size): Fake sequence.
        label_flip (bool): If True, then randomly flip the
            real and fake lable to confuse the discriminator.
        label_smooth (bool): Smooth the real label from exactly
            1 to random number sampled from uniform distribution
            from 0.8 to 1.0. Important: Do not applied label
            smoothing for fake labels.
        instance_noise (float): If other than 0, then add noise
            to training data.

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

def mmd_fn(real, fake, gamma=1):
    """ Maximum mean discrepancy (MMD) with
        radial basis function (RBF) kernel.
        Instead of returning MMD, this function return the
        -log(MMD) easy tracking.

    Args:
        real (batch, seq_len, feature_size): Real sequence.
        fake (batch, seq_len, feature_size): Fake sequence.
        gamma (float): RBF constant

    Returns:
        output (float): Computed MMD with RBF.
    """
    Br, Bf = real.shape[0], fake.shape[0]
    real = real.reshape(Br, -1)
    fake = fake.reshape(Bf, -1)

    Krr = (-gamma * torch.cdist(real, real).pow(2)).exp().sum() - Br
    Krf = (-gamma * torch.cdist(real, fake).pow(2)).exp().sum()
    Kff = (-gamma * torch.cdist(fake, fake).pow(2)).exp().sum() - Bf

    output = -((1/(Br*(Br-1)))*Krr - (2/(Br*Bf))*Krf + (1/(Bf*(Bf-1)))*Kff).abs().log()
    return output

def plot_generated(G, params, device):
    """ Plot a sample of generated data for tracking.

    Args:
        G (nn.Module): Generator.
        params (dictionary): Parameter dictionary.
        device : device.

    Returns:
        None
    """
    # Generate fake data:
    G = G.to(device)
    G.eval()
    noise = torch.randn((8, params["window_size"], params["noise_size"]), device=device)
    condition_same = torch.rand((1, 1, 1)).expand(4, params["window_size"], -1).to(device)
    condition_diff = torch.rand((4, 1, 1)).expand(-1, params["window_size"], -1).to(device)
    condition = torch.cat((condition_same, condition_diff))
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

def train_gan(G=None, D=None, history=None, trainloader=None, device="cpu", params=None):
    """ Training loop for GAN model.
        During training, a version of the generator will be saved to
        directory "saved_models" at the end of each epoch.

    Args:
        G (nn.Module): Generator.
        D (nn.Module): Discriminator.
        history (dictionary): Training log dictionary.
        trainloader (DataLoader): Train loader.
        device: device.
        params (dictionary): Hyperparameter dictionary.

    Returns:
        G (nn.Module): Trained generator.
        D (nn.Module): Trained discriminator.
        history (dictionary): Training log dictionary.
    """
    total_steps = math.ceil(len(trainloader.dataset) / params["batch_size"])
    new_line = total_steps // 2
    os.mkdir("./saved_models")

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
            "D_fake": [],
            "mmd": []
        }

    # Optimizer:
    optimG = optim.Adam(G.parameters(), lr=params["G_lr"], betas=(params["momentum"], 0.999))
    optimD = optim.Adam(D.parameters(), lr=params["D_lr"], betas=(params["momentum"], 0.999))

    # MMD preparation:
    batch = trainloader.dataset[torch.randperm(len(trainloader.dataset))[:64]]
    mmd_real = batch[0].to(device)

    # Main training loop:
    for epoch in range(params["max_epochs"]):
        for i, (sequence, condition) in enumerate(trainloader):
            G.train(); D.train()
            #################################################################
            # Train Discriminator: maximize log(D(x)) + log(1 - D(G(x)))    #
            #################################################################
            G.zero_grad(); D.zero_grad()
            # Form real and fake batch separately:
            noise = torch.randn((condition.shape[0], params["window_size"], params["noise_size"]), device=device)
            condition = condition.expand(-1, params["window_size"], -1).to(device)
            fake = G(noise, condition)
            real = sequence.to(device)
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

            #################################################################
            # Train Generator: maximize log(D(G(z)))                        #
            #################################################################
            G.zero_grad(); D.zero_grad()
            # Generate fake batch:
            fake = G(noise, condition)
            # Forward real and fake batch through D:
            lossG = generator_loss(D, fake, condition, params["instance_noise"])
            # Backpropagation:
            lossG.backward()
            # Update D:
            optimG.step()

            #################################################################
            # Compute MMD to quantitatively access generator                #
            #################################################################
            mmd_noise = torch.randn((64, params["window_size"], params["noise_size"]), device=device)
            mmd_condition = torch.rand((64, 1, 1)).expand(-1, params["window_size"], -1).to(device)
            with torch.no_grad():
                mmd_fake = G(mmd_noise, mmd_condition)
            mmd = mmd_fn(mmd_real, mmd_fake, params["gamma"])

            #################################################################
            # Print Training Progress                                       #
            #################################################################
            # Record training progress:
            history["lossD"].append(lossD.item())
            history["lossG"].append(lossG.item())
            history["D_real"].append(avg_pred_real.item())
            history["D_fake"].append(avg_pred_fake.item())

            # Print out training progress (on same line):
            stats = '[%3d/%3d][%3d/%3d]\tLossD: %.4f, LossG: %.4f, D_Real: %.4f, D_fake: %.4f, mmd: %f' \
                    % (epoch+1, params["max_epochs"], i, total_steps, lossD.item(), lossG.item(), \
                      avg_pred_real.item(), avg_pred_fake.item(), mmd.item())
            print('\r' + stats, end="", flush=True)

            # Print out training progress (on different line):
            if (i % new_line == 0):
                print('\r' + stats)

        # Visually tracking th quality of generated data:
        plot_generated(G, params, device)
        # Track best generator:
        history["mmd"].append(mmd.item())
        # Saved generator:
        torch.save(G.state_dict(),
                   "./saved_models/G_index_%s_epoch_%s_noise_%s_hidden_%s_feature_%s_layer_%s_drop_%s_window_%s.pth" %
                   (params["index"],
                    epoch+1,
                    params["noise_size"],
                    params["hidden_size"],
                    params["feature_size"],
                    params["num_layers"],
                    int(params["dropout"]*100),
                    params["window_size"]))

    # Done:
    return G, D, history

def train_rul(mode, trainloader, validloader, params, device="cpu", generator=None):
    """ Training loop for RUL prediction model.
        There are 3 training modes:
            If mode is "real", then train with only real data.
            If mode is "both", then train with both real and fake data.
            If mode is "fake", then train with only fake data.

    Args:
        mode (string): one in "real", "both", "fake".
        trainloader (DataLoader): Train dataloader.
        validloader (DataLoader): Validation dataloader.
        params (dictionary): Hyperparameter dictionary.
        device: device.
        generator (torch.Module): Trained generator.

    Returns:
        model (nn.Module): Trained RUL prediction model.
        history (dictionary): Training log dictionary.
    """
    # Initialize model:
    model = SimpleGRU(params["feature_size"],
                      params["hidden_size"],
                      params["num_layers"],
                      params["dropout"],
                      params["bidirectional"])
    model.to(device)
    # Initialize optimizer:
    optimizer = optim.Adam(model.parameters(),
                           lr=params["lr"],
                           weight_decay=params["weight_decay"])
    # Initialize learning rate scheduler:
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params["scheduler_factor"],
                                  patience=params["scheduler_patience"],
                                  verbose=False)
    # Initialize criterion:
    criterion = nn.MSELoss()

    # Training history:
    history = {
        'train_mse': [],
        'valid_mse': []
    }

    #Create trainer & evaluator:
    metrics = {
        'mse': Loss(criterion),
    }

    if mode == "real":
        trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    elif mode == "both":
        def both_train(engine, batch):
            # Generate fake data:
            noise = torch.randn((params["batch_size"]//8, params["window_size"], params["noise_size"])).to(device)
            condition = torch.FloatTensor(np.random.uniform(size=(params["batch_size"]//8, 1, 1))) \
                             .expand(-1, params["window_size"], -1) \
                             .to(device)
            with torch.no_grad():
                fake = generator(noise, condition)
            fake_inputs, fake_targets = fake, condition[:, :1, :1]
            # Forward + Backward + Update:
            real_inputs, real_targets = batch[0].to(device), batch[1].to(device)
            inputs = torch.cat((real_inputs, fake_inputs))
            targets = torch.cat((real_targets, fake_targets))
            optimizer.zero_grad()
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            return loss.item()
        trainer = Engine(both_train)
    elif mode == "fake":
        def fake_train(engine, batch):
            # Generate fake data:
            noise = torch.randn((params["batch_size"], params["window_size"], params["noise_size"])).to(device)
            condition = torch.FloatTensor(np.random.uniform(size=(params["batch_size"], 1, 1))) \
                             .expand(-1, params["window_size"], -1) \
                             .to(device)
            with torch.no_grad():
                inputs = generator(noise, condition)
            targets = condition[:, :1, :1]
            # Forward + Backward + Update:
            optimizer.zero_grad()
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            return loss.item()
        trainer = Engine(fake_train)
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    # Log training & validation results:
    @trainer.on(Events.ITERATION_COMPLETED(every=params["log_train_val"]))
    def log_results(trainer):
        evaluator.run(validloader)
        s = trainer.state
        m = evaluator.state.metrics
        history['train_mse'].append(s.output)
        history['valid_mse'].append(m['mse'])
        """print("Epoch [%d/%d] train_mse: %f\tval_mse: %f" \
              % (s.epoch, s.max_epochs, s.output, m['mse']), flush=True)"""

    # Learning rate schedule:
    @evaluator.on(Events.COMPLETED)
    def reduce_lr_on_plateau(evaluator):
        scheduler.step(evaluator.state.metrics['mse'])

    # Early Stopping - Tracking Validation Loss:
    def score_function(engine):
        return -engine.state.metrics['mse']
    earlystopper = EarlyStopping(patience=params["earlystop_patience"], score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, earlystopper)

    # Model Checkpoint:
    train_day = date.today().strftime("%d.%m.%y")
    checkpointer = ModelCheckpoint(f'./saved_models/{train_day}',
                                   "",
                                   score_function=score_function,
                                   score_name="mse",
                                   n_saved=1,
                                   global_step_transform=global_step_from_engine(trainer),
                                   create_dir=True,
                                   require_empty=False)
    evaluator.add_event_handler(Events.COMPLETED, checkpointer, {mode: model})

    # Run trainer
    trainer.run(trainloader, max_epochs=params["max_epochs"])

    return model, history
