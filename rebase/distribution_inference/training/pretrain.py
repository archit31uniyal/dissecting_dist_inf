import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
from model import ResNet


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

BATCH_SIZE = 256
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
NUM_WORKERS = 4

custom_transforms = transforms.Compose([
    transforms.CenterCrop((160, 160)),
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_dataloaders_celeba(batch_size, num_workers=0,
                           train_transforms=None,
                           test_transforms=None,
                           download=True):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()
        
    get_smile = lambda attr: attr[31]

    train_dataset = datasets.CelebA(root='.',
                                    split='train',
                                    transform=train_transforms,
                                    target_type='attr',
                                    target_transform=get_smile,
                                    download=download)

    valid_dataset = datasets.CelebA(root='.',
                                    split='valid',
                                    target_type='attr',
                                    target_transform=get_smile,
                                    transform=test_transforms)

    test_dataset = datasets.CelebA(root='.',
                                   split='test',
                                   target_type='attr',
                                   target_transform=get_smile,
                                   transform=test_transforms)


    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    valid_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    return train_loader, valid_loader, test_loader


train_loader, valid_loader, test_loader = get_dataloaders_celeba(
    batch_size=BATCH_SIZE,
    train_transforms=custom_transforms,
    test_transforms=custom_transforms,
    download=False,
    num_workers=4)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 3

# Architecture
num_features = 128*128
num_classes = 2

torch.manual_seed(random_seed)
model = ResNet(out_channels = num_classes)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    
def train(model, num_epochs, optimizer,):
    start_time = time.time()
    for epoch in range(num_epochs):
        
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            features = features.to(device)
            targets = targets.to(device)
                
            ### FORWARD AND BACK PROP
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' 
                    %(epoch+1, num_epochs, batch_idx, 
                        len(train_loader), cost))

            

        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            print('Epoch: %03d/%03d | Accuracy Train: %.3f%% | Accuracy Valid: %.3f%%' % (
                epoch+1, num_epochs, 
                compute_accuracy(model, train_loader),
                compute_accuracy(model, valid_loader)))
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    torch.save(model, "/p/compressionleakage/logs/Compressed/compression_cv/models/celeba/pretrained_resnet50_celeba.pt")    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

train()

with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))

for batch_idx, (features, targets) in enumerate(test_loader):
    features = features
    targets = targets
    break

model.eval()
logits, probas = model(features.to(device)[3, None])
print('Probability Smile %.2f%%' % (probas[0][1]*100))

