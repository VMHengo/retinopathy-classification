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
    
    model.train()
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
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    lr = 0.001 #learning rate

    task = "multi-label, binary-class"
    device = torch.device('cpu')
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders('retinamnist')
    model = ResNet18Binary().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizers = ( optim.Adam(model.parameters(), lr=lr), # Adaptive Moment Estimation
                   optim.SGD(model.parameters(), lr=lr, momentum=0.9) ) # Stochastic Gradient Descent

    for optimizer in optimizers:
        for epoch in range(NUM_EPOCHS):
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
            
            # print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
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
        print(f"Final Train Loss: {train_loss:.4f}, Final Train Acc: {train_acc:.4f}")
        print(f"Final Val Loss: {val_loss:.4f}, Final Val Acc: {val_acc:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}, Final Test Acc: {test_acc:.4f}")
        print("-------------------------------------------------------------")
        print("\n")


if __name__ == "__main__":
    train()