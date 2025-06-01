import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd
import os

from utils import save_prediction_as_image

class IntensityNet(nn.Module):
    def __init__(self):
        super(IntensityNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 60 * 36),
        )

    def forward(self, x):
        x = self.fc(x)     # [batch_size, 60 * 36]
        x = x.view(-1, 1, 60, 36)  # [batch_size, channels=1, H=60, W=36]
        return x

def train(model: IntensityNet, train_loader: DataLoader, criterion, optimizer, device, epoch)->float: 
    save_root = 'Final Project\Intensity_Generator\data_split\prediction\\train'
    model.train()
    total_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader)) # batch_size = 32

    for i, (target_matrix, features, base_name) in loop:
        # move data to device (where model is located)
        target_matrix = target_matrix.to(device)
        features = features.to(device)

        # forward
        outputs = model(features) # [batch_size, channels=1, H=60, W=36]
        loss = criterion(outputs, target_matrix) # criterion = nn.SmoothL1Loss()
        total_loss += loss.item()

        # save predictions
        if epoch == 9:
            for j in range(outputs.size(0)):
                filename = base_name[j]
                filename = f"{filename}.jpg"
                save_path = os.path.join(save_root, filename)
                save_prediction_as_image(outputs[j], save_path)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # tqdm display
        loop.set_description(f"Training")
        loop.set_postfix(loss=loss.item()) # display loss (should decrease through training)

    avg_loss = total_loss / len(train_loader)

    return avg_loss

def validate(model: IntensityNet, val_loader: DataLoader, criterion, device, epoch)->Tuple[float, float]:
    save_root = 'Final Project\Intensity_Generator\data_split\prediction\\train'
    model.eval()
    total_loss = 0.0
    total_size = 0
    correct_size = 0 # percentage of "good enough" predictions

    loop = tqdm(enumerate(val_loader), total=len(val_loader)) # batch_size = 32
    with torch.no_grad():
        for i, (target_matrix, features, base_name) in loop:
            # move data to device (where model is located)
            target_matrix = target_matrix.to(device)
            features = features.to(device)

            # forward
            outputs = model(features) # [batch_size, channels=1, H=60, W=36]
            loss = criterion(outputs, target_matrix) # criterion = nn.SmoothL1Loss()
            total_loss += loss.item()

            # validation
            total_size += outputs.size(0) # collect batch size
            for j in range(outputs.size(0)):
                if epoch == 9:
                    filename = base_name[j]
                    filename = f"{filename}.jpg"
                    save_path = os.path.join(save_root, filename)
                    save_prediction_as_image(outputs[j], save_path)

                if criterion(target_matrix[j], outputs[j]).item() < 0.1:
                    correct_size += 1

            # tqdm display
            loop.set_description(f"Validating")
            loop.set_postfix(loss=loss.item()) # display loss

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_size / total_size
    return avg_loss, accuracy

def test(model: IntensityNet, test_loader: DataLoader, criterion, device):
    save_root = 'Final Project\Intensity_Generator\data_split\prediction\\test'
    model.eval()
    
    total_loss = 0.0
    total_size = 0
    correct_size = 0 # percentage of "good enough" predictions

    loop = tqdm(enumerate(test_loader), total=len(test_loader)) # batch_size = 32
    with torch.no_grad():
        for i, (target_matrix, features, base_name) in loop:
            # move data to device (where model is located)
            target_matrix = target_matrix.to(device)
            features = features.to(device)

            # forward
            outputs = model(features) # [batch_size, channels=1, H=60, W=36]
            loss = criterion(outputs, target_matrix) # criterion = nn.SmoothL1Loss()
            total_size += outputs.size(0) # collect batch size
            total_loss += loss.item()

            # save predictions
            for j in range(outputs.size(0)):
                filename = base_name[j]
                filename = f"{filename}.jpg"
                save_path = os.path.join(save_root, filename)
                save_prediction_as_image(outputs[j], save_path)
                
                if criterion(target_matrix[j], outputs[j]).item() < 0.05:
                    correct_size += 1

            # tqdm display
            loop.set_description(f"Testing")
            loop.set_postfix(loss=loss.item()) # display loss

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_size / total_size
    return avg_loss, accuracy
