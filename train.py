import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np

import medmnist
from medmnist import INFO, Evaluator
from dataset_loader import get_dataloaders
from model import ResNet18Binary


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def load_info(data_flag:str) -> (str, int, int):
    # load retinamnist variables
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = 2 # reduce classes from 0-4 (low to high) to 0-1 (neglible or severe)
    return task, n_channels, n_classes

def run_epoch(dataloader, model, criterion, device, optimizer=None):
    training_active = optimizer is not None

    total_correct = 0
    total_loss = 0.0
    average_loss = 0
    average_acc = 0
    total_inputs = 0
    
    if training_active:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(training_active):
        for inputs, targets in tqdm(dataloader):

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            targets = targets.to(torch.float32).unsqueeze(1)
            loss = criterion(outputs, targets)
            preds = (outputs >= 0).long() # interpret logit values as binary predictions
            correct_pred_vect = (preds == targets) # tensor([[bool], [bool], ...])
            total_correct += correct_pred_vect.sum().item() # sum of correct vectors as tesnor -> extract int with .item()
            
            total_inputs += inputs.shape[0]
            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size

            if training_active:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    average_loss = total_loss / total_inputs
    average_acc = total_correct / total_inputs
    
    return average_loss, average_acc

def train():
    MAX_NUM_EPOCHS = 20
    lr = 0.001 #learning rate

    device = get_device()
    print(f"Using device: {device}")
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders('retinamnist')
    criterion = nn.BCEWithLogitsLoss()

    optimizer_configs = [
        ("Adam", optim.Adam, {"lr": lr}), # adaptive moment estimation
        ("SGD", optim.SGD, {"lr": lr, "momentum": 0.9}), # stochastic gradient descent
    ]

    # test training on different optimizers
    for optimizer_name, optimizer_class, optimizer_kwargs in optimizer_configs:
        model = ResNet18Binary().to(device)
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        # for reducing overfitting
        best_val_loss = float("inf")
        patience = 2
        epochs_without_improvement = 0

        for epoch in range(MAX_NUM_EPOCHS):
            train_loss, train_acc = run_epoch(
                dataloader=train_dataloader,
                model=model,
                criterion=criterion,
                device=device,
                optimizer=optimizer
            )
            val_loss, val_acc = run_epoch(
                dataloader=val_dataloader,
                model=model,
                criterion=criterion,
                device=device
            )
            
            min_improvement = 0.01
            if (val_loss + min_improvement) < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

                # stop training if val_loss stops improving/decreasing
                if epochs_without_improvement >= patience:
                    break
            
            # print progress each epoch
            # print(f"Epoch {epoch+1}/{MAX_NUM_EPOCHS}")
            # print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            # print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # run on test set to see final performance
        test_loss, test_acc = run_epoch(
            dataloader=test_dataloader,
            model=model,
            criterion=criterion,
            device=device
        )

        print("-------------------------------------------------------------")
        print(f"Optimizer: {optimizer.__class__.__name__}")
        print(f"Stopped at epoch: {epoch+1}/{MAX_NUM_EPOCHS}")
        print(f"Final Train Loss: {train_loss:.4f}, Final Train Acc: {train_acc:.4f}")
        print(f"Final Val Loss: {val_loss:.4f}, Final Val Acc: {val_acc:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}, Final Test Acc: {test_acc:.4f}")
        print("-------------------------------------------------------------")
        print("\n")


if __name__ == "__main__":
    train()
