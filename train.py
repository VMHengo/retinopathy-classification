import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import copy
import os

import medmnist
from medmnist import INFO, Evaluator
from dataset_loader import get_dataloaders
from model import ResNet18Binary


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if os.getenv("USE_DIRECTML") != "1":
        return torch.device("cpu")

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

def run_epoch(dataloader, model, criterion, device, optimizer=None, threshold=0.5):
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
            probabilities = torch.sigmoid(outputs)
            preds = (probabilities >= threshold).long()
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


def find_best_threshold(dataloader, model, device, thresholds=None):
    if thresholds is None:
        thresholds = [i / 100 for i in range(10, 91, 5)]

    model.eval()
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs).cpu()

            targets = targets.float().unsqueeze(1).cpu()
            all_probabilities.append(probabilities)
            all_targets.append(targets)

    all_probabilities = torch.cat(all_probabilities)
    all_targets = torch.cat(all_targets)

    best_threshold = 0.5
    best_acc = 0.0

    for threshold in thresholds:
        preds = (all_probabilities >= threshold).long()
        acc = (preds == all_targets.long()).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_threshold, best_acc

def train():
    MAX_NUM_EPOCHS = 20

    device = get_device()
    print(f"Using device: {device}")
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    train_dataloader, test_dataloader, val_dataloader = get_dataloaders('retinamnist')
    criterion = nn.BCEWithLogitsLoss()

    optimizer_configs = [
        ("Adam", optim.Adam, {"lr": 0.001, "weight_decay": 1e-4}), # adaptive moment estimation
        ("SGD", optim.SGD, {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4}), # stochastic gradient descent
    ]

    # test training on different optimizers
    for optimizer_name, optimizer_class, optimizer_kwargs in optimizer_configs:
        model = ResNet18Binary().to(device)
        best_model_state = copy.deepcopy(model.state_dict())
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        # for reducing overfitting
        best_val_loss = float("inf")
        patience = 5
        min_improvement = 0.001
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
            
            if (val_loss + min_improvement) < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

                # stop training if val_loss stops improving/decreasing
                if epochs_without_improvement >= patience:
                    break
            
            # print progress each epoch
            # print(f"Epoch {epoch+1}/{MAX_NUM_EPOCHS}")
            # print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            # print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Evaluate the best validation model
        model.load_state_dict(best_model_state)

        best_threshold, best_threshold_val_acc = find_best_threshold(
            dataloader=val_dataloader,
            model=model,
            device=device
        )

        checkpoint = {
            "model_state_dict": best_model_state,
            "optimizer_name": optimizer_name,
            "best_val_loss": best_val_loss,
            "best_threshold": best_threshold,
            "best_threshold_val_acc": best_threshold_val_acc,
        }
        # export the best model
        torch.save(checkpoint, weights_dir / f"best_model_{optimizer_name}.pt")

        final_train_loss, final_train_acc = run_epoch(
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            device=device,
            threshold=best_threshold
        )

        final_val_loss, final_val_acc = run_epoch(
            dataloader=val_dataloader,
            model=model,
            criterion=criterion,
            device=device,
            threshold=best_threshold
        )

        test_loss_fixed_threshold, test_acc_fixed_threshold = run_epoch(
            dataloader=test_dataloader,
            model=model,
            criterion=criterion,
            device=device,
            threshold=0.5
        )

        test_loss_tuned_threshold, test_acc_tuned_threshold = run_epoch(
            dataloader=test_dataloader,
            model=model,
            criterion=criterion,
            device=device,
            threshold=best_threshold
        )

        print("-------------------------------------------------------------")
        print(f"Optimizer: {optimizer_name}")
        print(f"Stopped at epoch: {epoch+1}/{MAX_NUM_EPOCHS}")
        print(f"Best Threshold: {best_threshold:.2f}, Val Acc at Threshold: {best_threshold_val_acc:.4f}")
        print(f"Final Train Loss: {final_train_loss:.4f}, Final Train Acc: {final_train_acc:.4f}")
        print(f"Final Val Loss: {final_val_loss:.4f}, Final Val Acc: {final_val_acc:.4f}")
        print(f"Final Test Loss at 0.50 Threshold: {test_loss_fixed_threshold:.4f}, Final Test Acc at 0.50 Threshold: {test_acc_fixed_threshold:.4f}")
        print(f"Final Test Loss at Tuned Threshold: {test_loss_tuned_threshold:.4f}, Final Test Acc at Tuned Threshold: {test_acc_tuned_threshold:.4f}")
        print("-------------------------------------------------------------")
        print("\n")


if __name__ == "__main__":
    train()
