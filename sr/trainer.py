#!/usr/bin/env python3

"""Usage:   trainer.py --train=train_path --valid=valid_path --log_dir=log_dir --num_epochs=epochs
                       --architecture=arch --loss_list=loss1:loss2:loss3:loss4  --act=act --resume=model_to_resume [--debug_input_pics]
            trainer.py --help | -help | -h

Train the requested model.
Arguments:
  train         a directory with training images/ direcrtories/ numpy
  output        a directory for validation images/ directories/ numpy
  log_dir       directory for storing training logs
  num_epochs    number of epochs
  architecture  the architecture to train unet or axial
  loss_list     combination of losses to use. : separated list
  act activations can be relu, elu, or leakyrelu, prelu, FOR EDSR ONLY
  resume        path for model to resume training. Specify 'None' if new training
  --debug_input_pics  If we want to save input pics for debugging
Options:
  -h --help -h
"""

from pathlib import Path
import os
import shutil
from time import time
import datetime

import torch
import torch.optim as optim
import torch.nn

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from docopt import docopt

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from edsr import EDSR
from dataset import SrDataset

BATCH_SIZE = {"edsr_16_64": 8, "edsr_8_256": 16,
              "edsr_16_256": 8, "edsr_32_256": 8,}
LR = {"edsr_16_64": 0.0001, "edsr_8_256": 0.0001,
        "edsr_16_256": 0.0001, "edsr_32_256": 0.0001}


def create_dataset(path, architecture, lognorm=False):
    """

    Parameters
    ----------
    path: path to data directory
    lognorm: Is log normalization used?

    Returns
    -------
    Loaded dataset

    """
    return SrDataset(path, lognorm=lognorm)

def model_save(train_model, train_model_path):
    """

    Parameters
    ----------
    train_model: model state dictionary
    train_model_path: model path

    Returns
    -------

    """
    model_path = Path(train_model_path)
    model_folder = model_path.parent
    if not model_folder.is_dir():
        os.makedirs(model_folder)
    torch.save(train_model.state_dict(), model_path)


def training(training_generator, validation_generator, device, log_dir,
             architecture, losses, num_epochs, debug_pics, act, resume):
    """

    Parameters
    ----------
    training_generator: contains training data
    validation_generator: contains validation data
    device: Cuda Device
    log_dir: The log directory for storing logs
    architecture: The architecture to be used unet or axial
    losses: list of losses
    num_epochs:   The number of epochs
    debug_pics: True if we want to save pictures in input_pics
    act: activation function to be used
    resume: path to model for training resumption. None for new model
    Returns
    -------

    """
    timestamp = f'{datetime.datetime.now().date()}-{datetime.datetime.now().time()}' 
    save_model_path = (Path(__file__).parent/ "saved_models").resolve()
    writer = SummaryWriter(log_dir)
    if not save_model_path.is_dir():
        os.makedirs(save_model_path)
    save_model_path = str(save_model_path)
    # use architecture to create model
    lr = LR[architecture]
    if architecture == "edsr_16_64":
        model = EDSR(n_resblocks=16, n_feats=64, scale=1, act=act)
    elif architecture == "edsr_8_256":
        model = EDSR(n_resblocks=8, n_feats=256, scale=1, act=act)
    elif architecture == "edsr_16_256":
        model = EDSR(n_resblocks=16, n_feats=256, scale=1, act=act)
    elif architecture == "edsr_32_256":
        model = EDSR(n_resblocks=32, n_feats=256, scale=1, act=act)
    
    model.to(device)
    if resume is not None:
        model.load_state_dict(torch.load(resume))
        print("Model loaded. Resuming training")
    else:
        summary(model, (1, 256, 256), batch_size=1, device="cuda")
    max_epochs = num_epochs
    
    # set up dictionary of losses and loss weights
    criterion = dict()
    percep_criterion =False
    loss_weights = dict()
    for loss in losses:
        if loss == "L1":
            criterion["L1"] = torch.nn.L1Loss()
            loss_weights["L1"] = 1.0
        elif loss == "MSE":
            criterion["MSE"] = torch.nn.MSELoss(reduction='mean')
            loss_weights["MSE"] = 0.1
        elif loss == "SSIM":
            criterion["SSIM"] = SSIM()
            loss_weights["SSIM"] = 30.0
        elif loss == "perceptual":
            # Perceptual loss is used after 2/3 rd of the epochs
            print("Perceptual loss starts getting used later")
            percep_criterion = True
            loss_weights["perceptual"] = 0.1
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # learning rate scheduler
    # if architecture == "edsr_16_64":
    #    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    best_valid_loss = float("inf")
    best_epoch = 0
    step = 0
    totiter = len(training_generator)
    valiter = len(validation_generator)
    # For saving the images read in to perfrom debugging
    if debug_pics:
        input_save_path = Path("input_pics").resolve()
        if input_save_path.is_dir():
            shutil.rmtree(input_save_path)
        os.makedirs(input_save_path)
    for epoch in range(max_epochs):
        start_time = time()
        train_loss = valid_loss = 0.0
        loss_values = dict()
        val_loss_values = dict()
        model.train()
        if percep_criterion and epoch == (max_epochs * 2) // 3:
            criterion["perceptual"] = PerceptualLoss()
        loss_train_list = []
        step += 1
        # Main training loop for this epoch
        for batch_idx, data in tqdm(enumerate(training_generator), total=totiter):
            model.train(True)
            x_train = data["lr"]
            y_train = data["hr"]
            stat = data["stats"]
            mean, sigma = stat["mean"], stat["std"]
            x_train, y_train, mean, sigma = (
                x_train.to(device),
                y_train.to(device),
                mean.to(device),
                sigma.to(device),
            )
            # For saving the images read in to do debugging
            if debug_pics:
                x_np = x_train.cpu().numpy()
                y_np = y_train.cpu().numpy()
                for i in range(x_np.shape[0]):
                    filename = data["file"][i]
                    x_rescale = x_np[i] * stat["std"][i].numpy() + stat["mean"][i].numpy()
                    y_rescale = y_np[i] * stat["std"][i].numpy() + stat["mean"][i].numpy()

                    save_plots = np.hstack([x_rescale.reshape(x_rescale.shape[1], -1), y_rescale.reshape(y_rescale.shape[1], -1)])
                    save_plots = np.clip(save_plots, stat["min"][i].numpy(), stat["max"][i].numpy())
                    filename = os.path.join(f"{input_save_path}/{filename}.tiff")
                    plt.imsave(filename, save_plots, cmap='gray')
            
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                # Computation for tracking and summing all losses
                with torch.set_grad_enabled(True):
                    y_pred = model(x_train)
                    loss_train = 0.0
                    for key in criterion.keys():
                        if batch_idx == 0:
                            loss_values[key] = 0.0
                        this_loss = criterion[key](y_pred, y_train)
                        loss_values[key] = loss_values[key] + (
                            (1 / (batch_idx + 1)) * (this_loss.data - loss_values[key]))
                        loss_train += (loss_weights[key] * this_loss)
                    train_loss = train_loss + (
                        (1 / (batch_idx + 1)) * (loss_train.data - train_loss)
                    )
                    loss_train.backward()
                    optimizer.step()
        
        del x_train, y_train, mean, sigma, loss_train_list
        torch.cuda.empty_cache()

        # Main validation loop for this epoch
        # if architecture == "edsr_16_64":
        #    scheduler.factor = 1 + (epoch / max_epochs) ** 0.9
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(validation_generator), total=valiter):
                model.train(False)
                x_valid = data["lr"]
                y_valid = data["hr"]
                # loss computation on validation set
                x_valid, y_valid = x_valid.to(device), y_valid.to(device)
                y_pred = model(x_valid)
                loss_valid = 0.0
                weight_sum = 0.0
                for key in criterion.keys():
                    if batch_idx == 0:
                        val_loss_values[key] = 0.0
                    this_loss = criterion[key](y_pred, y_valid)
                    val_loss_values[key] = val_loss_values[key] + (
                        (1 / (batch_idx + 1)) * (this_loss.data - val_loss_values[key]))
                    weight_sum += loss_weights[key]
                    loss_valid += (loss_weights[key] * this_loss)

                loss_valid = loss_valid / weight_sum
                valid_loss = valid_loss + (
                    (1 / (batch_idx + 1)) * (loss_valid.data - valid_loss)
                )
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", valid_loss, epoch)
        # if architecture == "edsr_16_64":
            # calling scheduler based on valid loss
        #     scheduler.step(valid_loss)

        del x_valid, y_valid
        memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        print(
                "\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} in {:.1f} seconds. [lr:{:.8f}][max mem:{:.0f}MB]".format(
                epoch,
                train_loss,
                valid_loss,
                time() - start_time,
                optimizer.param_groups[0]["lr"],
                memory,
            )
        )
        print("Best epoch: " + str(best_epoch) + " Best Val loss: " + str(best_valid_loss))
        print("Training losses: ", loss_values)
        print("Validation losses: ", val_loss_values)
        # Save best validation epoch model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            model_save(
                model, f"{save_model_path}/{architecture}/{timestamp}_best_model.pt",
            )

        model_save(model, f"{save_model_path}/{architecture}/{timestamp}_model.pt")
        torch.cuda.empty_cache()
    
    writer.flush()
    writer.close()

def process(arguments):
    """

    Parameters
    ----------
    arguments: all the argumennts passed to trainer.py

    Returns
    -------

    """

    # Read and assign parameters to variables
    train_path = Path(arguments["--train"])
    valid_path = Path(arguments["--valid"])
    log_dir = Path(arguments["--log_dir"])
    architecture = arguments["--architecture"]
    loss_list = arguments["--loss_list"]
    losses = loss_list.split(':')
    num_epochs = int(arguments["--num_epochs"]) 
    debug_pics = arguments["--debug_input_pics"]
    act = arguments["--act"]
    resume = arguments["--resume"]
    if resume == 'None':
        resume = None
    parameters_train = {
        "batch_size": BATCH_SIZE[architecture],
        "shuffle": True,
        "num_workers": 6,
    }
    parameters_val = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 6,
    }


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Create training dataset for loading
    training_set = create_dataset(train_path, architecture)
    training_generator = torch.utils.data.DataLoader(training_set, **parameters_train)

    # Create validation dataset for loading
    validation_set = create_dataset(valid_path, architecture)
    validation_generator = torch.utils.data.DataLoader(validation_set, **parameters_val)
    training(training_generator, validation_generator, device, log_dir,
             architecture, losses, num_epochs, debug_pics, act, resume)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    process(arguments)
