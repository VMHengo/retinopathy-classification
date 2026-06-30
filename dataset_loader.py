import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

import medmnist
from medmnist import INFO, Evaluator


NUM_EPOCHS = 5
BATCH_SIZE = 128
lr = 0.001

def map_label(label):
    # Map RetinaMNIST labels 0-1 to 0 and labels 2-4 to 1.
    
    label = int(np.asarray(label).item())
    return 0 if label <= 1 else 1

# class to convert ordinal regression to binary classification
class BinaryRetinaDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        label = map_label(label)
        return image, torch.tensor(label, dtype=torch.long)

def get_datasets(data_flag:str, convert_to_binary=True) -> (data.Dataset, data.Dataset, data.Dataset):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    test_dataset = DataClass(split='test', transform=data_transform, download=True)
    val_dataset = DataClass(split='val', transform=data_transform, download=True)

    if convert_to_binary:
        # convert ordinal regression to binary classification
        train_dataset = BinaryRetinaDataset(train_dataset)
        test_dataset = BinaryRetinaDataset(test_dataset)
        val_dataset = BinaryRetinaDataset(val_dataset)

    return train_dataset, test_dataset, val_dataset

def get_dataloaders(data_flag:str) -> (data.DataLoader, data.DataLoader, data.DataLoader):
    train_dataset, test_dataset, val_dataset = get_datasets(data_flag, convert_to_binary=True)

    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader, val_dataloader

if __name__ == "__main__":
    train_dataset, test_dataset, val_dataset = get_datasets("retinamnist", convert_to_binary=False)
    print(train_dataset)
    print("===================")
    print(test_dataset)