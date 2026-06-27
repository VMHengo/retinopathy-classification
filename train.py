import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from dataset_loader import get_dataloaders, n_channels, n_classes
from model import Net


def load_info(data_flag:str) -> tuple(str, int, int):
    # load retinamnist variables
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = 2 # reduce classes from 0-4 (low to high) to 0-1 (neglible or severe)
    return task, n_channels, n_classes