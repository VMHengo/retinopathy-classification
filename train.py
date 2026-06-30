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


def train():
    NUM_EPOCHS = 3
    BATCH_SIZE = 128
    lr = 0.001

    task = "multi-label, binary-class"
    device = torch.device('cpu')
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders('retinamnist')
    model = ResNet18Binary().to(device)
    criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
    optimizers = ( optim.Adam(model.parameters(), lr=lr), 
                   optim.SGD(model.parameters(), lr=lr, momentum=0.9) )

    for optimizer in optimizers:
        for epoch in range(NUM_EPOCHS):
            train_correct = 0
            train_total = 0
            test_correct = 0
            test_total = 0
            
            model.train()
            for inputs, targets in tqdm(train_dataloader):
                # forward + backward + optimize
                optimizer.zero_grad()
                outputs = model(inputs)
                
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32).unsqueeze(1)
                    loss = criterion(outputs, targets)
                else:
                    targets = targets.squeeze().long()
                    loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()


# evaluation

def test(split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)
    
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

if __name__ == "__main__":
    train()
    print('==> Evaluating ...')
    test('train')
    test('test')