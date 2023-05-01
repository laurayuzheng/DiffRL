''' Noise model which is learned from real world data. Uses IDM as the car-following model backbone.
Author: Laura Zheng
'''

import os, glob
import numpy as np
import pandas as pd
import torch

from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import NGSimDataset

class MLP(nn.Module):
    def __init__(self):
        super(MLP).__init__()

        self.fc1 = nn.Linear(150, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class NoiseModel:

    def __init__(self, datadir, save_path="./weights/", lr=0.001, load_weights=None):
        self.datadir = datadir
        self.save_path = save_path

        self.net = MLP()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.curr_epoch = 0
        self.curr_loss = 0

        if load_weights:
            checkpoint = torch.load(load_weights)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.curr_epoch = checkpoint['epoch']
            self.curr_loss = checkpoint['loss']

    def train_model(self):
        pass


    def save_model(self, epoch, loss):

        torch.save({
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, self.save_path)


if __name__ == "__main__":

    dataset = NGSimDataset("./data/trajectories-0400-0415.csv")
