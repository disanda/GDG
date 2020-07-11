import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import torchvision

#-------------------one-channel dataset-------------

# data
transform1 = transforms.Compose(
    [transforms.Scale(size=(32, 32), interpolation=Image.BICUBIC),
     transforms.ToTensor(),
     transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)

def getDataloader(batch_size,use_gpu):
    train_loader = torch.utils.data.DataLoader(
    #dataset=dsets.FashionMNIST('data/FashionMNIST', train=True, download=True, transform=transform),
    #dataset=torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform),
    dataset=torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transform1),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=use_gpu,
    drop_last=True)
    return train_loader
    