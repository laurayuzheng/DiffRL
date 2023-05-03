''' Noise model which is learned from real world data. Uses IDM as the car-following model backbone.
Author: Laura Zheng
'''

import os, glob
import tqdm 
import numpy as np
import pandas as pd
import torch

from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import NGSimDataset

from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

CSV_PATHS = ["./data/trajectories-0400-0415.csv",
             "./data/trajectories-0500-0515.csv", 
             "./data/trajectories-0515-0530.csv"]

class MLP(nn.Module):
    def __init__(self, num_vehicles):

        super(MLP, self).__init__()

        self.fc1 = nn.Linear(num_vehicles*10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_vehicles)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class NoiseModel:

    def __init__(self, save_path="./weights/", lr=0.001, num_epochs=100, load_weights=None, max_vehicles=300):
        self.save_path = save_path

        self.net = MLP(max_vehicles)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.writer = SummaryWriter(log_dir="./runs", comment="noise_model")

        self.curr_epoch = 0
        self.curr_loss = 0
        self.total_epochs = num_epochs
        self.max_vehicles = 300

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if load_weights:
            checkpoint = torch.load(load_weights)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.curr_epoch = checkpoint['epoch']
            self.curr_loss = checkpoint['loss']
        
        self.dataset = NGSimDataset(CSV_PATHS, device=self.device, max_vehicles=max_vehicles)
        self.net.to(self.device)

    def train_model(self):
        
        self.net.train() 

        for param in self.net.parameters():
            param.requires_grad = True

        loss = 0 
        dataloader = DataLoader(self.dataset, batch_size=5000, 
                                pin_memory=False)
                                # pin_memory=(self.device == "cuda"))
        
        total_updates = self.total_epochs * len(dataloader)
        update_idx = 0

        for epoch in range(self.curr_epoch, self.total_epochs):
            
            with tqdm.tqdm(dataloader, unit="batch") as tepoch:

                for b_idx, (obs0, obs1, rand_indices, idxs) in enumerate(tepoch):
                    
                    tepoch.set_description(f"Epoch {epoch}")

                    # clear gradients
                    for param in self.net.parameters():
                        param.grad = None

                    obs0 = obs0.to(self.device, dtype=torch.float32)
                    obs1 = obs1.to(self.device, dtype=torch.float32)

                    rand_indices = rand_indices.to(self.device, dtype=torch.float32)

                    with autocast(enabled=(self.device=="cuda")):

                        noise = self.net(obs0) # forward pass 
                        
                        obs_t1_hats = self.dataset.sim.add_noise_and_forward(noise,idxs,rand_indices=rand_indices)

                        _loss = self.criterion(obs_t1_hats, obs1)
                        _loss.backward() 

                        self.optimizer.step()

                        self.writer.add_scalar("Loss/train", loss, update_idx)

                        update_idx += 1

                    loss += float(_loss.item()) 

                    self.scheduler.step()
                    # self.scheduler.step_update(num_updates=total_updates, metric=loss/(b_idx+1))

                    tepoch.set_postfix(loss=_loss.item())
            
            # self.scheduler.step(epoch+1, loss/len(dataloader))
        
        print("Training finished")
        self.save_model(epoch=self.total_epochs-1, loss=loss)

    def save_model(self, epoch, loss):

        torch.save({
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(self.save_path, 'final-epochs=%d.pt'%(self.total_epochs)))


if __name__ == "__main__":

    noise_model = NoiseModel(max_vehicles=100)
    noise_model.train_model()
