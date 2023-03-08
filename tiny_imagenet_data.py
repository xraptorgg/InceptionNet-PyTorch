"""
A script to prepare training and test DataLoader
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchsummary import summary
import matplotlib.pyplot as plt
import os



# dataset

def prepare_dataset(data_path):
    """
    Create PyTorch datasets.
    Args:
        data_path (path): Path to the Tiny ImageNet data.
    Return:
        training_data (datatset): Training dataset.
        val_data (datatset): Validation dataset.
    """

    training_path = f"{data_path}/train"
    val_path = f"{data_path}/val"

    with open(f"{val_path}/val_annotations.txt") as f:
        lines = f.readlines()
    
    val_dict = {}

    for line in lines:
        parts = line.strip().split('\t')
        val_dict[parts[0]] = parts[1]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    training_data = datasets.ImageFolder(root = training_path, transform = transform, target_transform = None)
    val_data = datasets.ImageFolder(root = val_path, transform = transform, target_transform = None)

    for i in range(len(val_data)):
        img_path, _ = val_data.imgs[i]
        img_name = os.path.basename(img_path)
        val_data.imgs[i] = (img_path, training_data.classes.index(val_dict[img_name]))

    return training_data, val_data


# dataloader

def prepare_dataloader(batch_size, training_data, val_data):
    """
    Create PyTorch dataloaders.
    Args:
        batch_size (int): Size of each batch of data.
        training_data (datatset): Training dataset.
        val_data (datatset): Validation dataset.
    Return:
        training_dataloader (DataLoader): Training dataloader.
        val_dataloader (DataLoader): Validation dataloader.
    """

    torch.manual_seed(1234)

    training_dataloader = DataLoader(dataset = training_data, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(dataset = val_data, batch_size = batch_size, shuffle = True)

    return training_dataloader, val_dataloader
