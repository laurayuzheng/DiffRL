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

from dataset import NGSimDataset, NGSimDatasetOffline, NGSimDatasetOfflineGraph

from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import GCN
from torch_geometric.loader import DataLoader as GraphLoader
from torch_geometric.utils import to_dense_batch

CSV_PATHS_TRAIN = ["./data/train_i80.csv",
            #  "./data/trajectories-0500-0515.csv", 
            #  "./data/trajectories-0515-0530.csv"
             ]

CSV_PATHS_TEST = ["./data/test_i80.csv",
            #  "./data/trajectories-0500-0515.csv", 
            #  "./data/trajectories-0515-0530.csv"
             ]

MAX_VEHICLES = 200

IS_GRAPH = True 

NAME = "may8_graph"

class MLP(nn.Module):
    def __init__(self, num_vehicles):

        super(MLP, self).__init__()

        self.fc1 = nn.Linear(num_vehicles*10, 200)
        self.fc2 = nn.Linear(200, 64)
        # self.fc3 = nn.Linear(200, 64)
        self.fc3 = nn.Linear(64, num_vehicles)

        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return self.fc3(x)


class NoiseModel:

    def __init__(self, save_path="./weights/", experiment="test", lr=1e-5, num_epochs=100, load_weights=None, max_vehicles=MAX_VEHICLES, batch_size=50, is_graph=False):
        
        os.makedirs(os.path.join(save_path, experiment), exist_ok=True)

        self.save_path = os.path.join(save_path, experiment)

        self.exp_name = experiment 

        os.makedirs(self.save_path, exist_ok=True)

        self.net = MLP(max_vehicles) if is_graph == False else GCN(in_channels=10, hidden_channels=64, num_layers=2, out_channels=1)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.writer = SummaryWriter(log_dir="./runs_graph=%s"%("1" if IS_GRAPH else "0"), comment="noise_model")
        self.batch_size = batch_size

        self.curr_epoch = 0
        self.curr_loss = 0
        self.total_epochs = num_epochs
        self.max_vehicles = max_vehicles

        self.is_graph = is_graph

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if load_weights:
            checkpoint = torch.load(load_weights)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.curr_epoch = checkpoint['epoch']
            self.curr_loss = checkpoint['loss']
        
        if is_graph: 
            self.dataset_train = NGSimDatasetOfflineGraph("./data/pt_unshuffled/train", CSV_PATHS_TRAIN, device=self.device, max_vehicles=max_vehicles, num_envs=1)
            self.dataset_val = NGSimDatasetOfflineGraph("./data/pt_unshuffled/test", CSV_PATHS_TEST, device=self.device, max_vehicles=max_vehicles, num_envs=1)
        else:
            self.dataset_train = NGSimDatasetOffline("./data/npy_unshuffled/train", CSV_PATHS_TRAIN, device=self.device, max_vehicles=max_vehicles, num_envs=1)
            self.dataset_val = NGSimDatasetOffline("./data/npy_unshuffled/test", CSV_PATHS_TEST, device=self.device, max_vehicles=max_vehicles, num_envs=1)
        
        self.net.to(self.device)

    def train_one_epoch(self, dataloader, epoch):
            
        self.net.train() 

        for param in self.net.parameters():
            param.requires_grad = True

        loss = 0.0

        with tqdm.tqdm(dataloader, unit="batch") as tepoch:

            for b_idx, (obs0, obs1, rand_indices, idxs, tdiff) in enumerate(tepoch):
                
                tepoch.set_description(f"Epoch {epoch}")

                # clear gradients
                for param in self.net.parameters():
                    param.grad = None

                obs0 = obs0.to(self.device)
                obs1 = obs1.to(self.device)
                
                # reshape batch to [batch_size, num_max_nodes, num_features]
                obs0_reshape, obs0_mask = to_dense_batch(obs0.x, obs0.x_batch, fill_value=0)

                rand_indices = rand_indices.to(self.device)

                with autocast(enabled=(self.device=="cuda")):
                    
                    self.optimizer.zero_grad()

                    noise = self.net(obs0_reshape, obs0.edge_index) # forward pass 

                    obs_t1_hats = self.dataset_train.sim.add_noise_and_forward(noise,idxs,rand_indices=rand_indices, tdiff=tdiff)

                    _loss = self.criterion(obs_t1_hats.flatten(), obs1.flatten())
                    _loss.backward() 

                    # self.writer.add_scalar("Loss/train/iter", _loss, b_idx + epoch*len(dataloader))

                    self.optimizer.step()

                loss += float(_loss.item()) 

                # self.scheduler.step()
                # self.scheduler.step_update(num_updates=total_updates, metric=loss/(b_idx+1))

                tepoch.set_postfix(loss=_loss.item())

            self.writer.add_scalar("Loss/train", loss/len(dataloader), epoch)

            return loss/len(dataloader)

    def eval_one_epoch(self, dataloader, epoch, best_v_loss):
        
        self.net.eval()

        for param in self.net.parameters():
            param.requires_grad = False

        loss = 0.0

        with tqdm.tqdm(dataloader, unit="batch") as tepoch:

            for b_idx, (obs0, obs1, rand_indices, idxs, tdiff) in enumerate(tepoch):
                
                tepoch.set_description(f"Epoch {epoch}")

                with torch.no_grad():

                    # obs0 = obs0.to(self.device, dtype=torch.float32)
                    # obs1 = obs1.to(self.device, dtype=torch.float32)
                    
                    obs0 = obs0.to(self.device)
                    obs1 = obs1.to(self.device)

                    obs0_reshape, obs0_mask = to_dense_batch(obs0.x, obs0.x_batch, fill_value=0)

                    rand_indices = rand_indices.to(self.device, dtype=torch.float32)

                    with autocast(enabled=(self.device=="cuda")):
                                                
                        noise = self.net(obs0_reshape, obs0.edge_index) # forward pass 

                        obs_t1_hats = self.dataset_val.sim.add_noise_and_forward(noise,idxs,rand_indices=rand_indices, tdiff=tdiff)

                        _loss = self.criterion(obs_t1_hats.flatten(), obs1.flatten())
                    
                loss += float(_loss.item()) 

                tepoch.set_postfix(loss=_loss.item())

            self.writer.add_scalar("Loss/val", loss/len(dataloader), epoch)
            
            if loss/len(dataloader) < best_v_loss:
                self.save_model(epoch, loss/len(dataloader), name="best_vloss")

            return loss/len(dataloader)

    def train_model(self):

        train_loss_list = []
        val_loss_list = []

        if self.is_graph:
            dataloader = GraphLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, follow_batch=['x'])
            eval_dataloader = GraphLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, follow_batch=['x'])
        
        else:
            dataloader = DataLoader(self.dataset_train, batch_size=self.batch_size, 
                                    pin_memory=False, shuffle=True)
            eval_dataloader = DataLoader(self.dataset_val, batch_size=self.batch_size, 
                                    pin_memory=False, shuffle=True)
                                # pin_memory=(self.device == "cuda"))
        
        # total_updates = self.total_epochs * len(dataloader)
        # update_idx = 0

        best_v_loss = 1_000_000

        for epoch in range(self.curr_epoch, self.total_epochs):
            
                train_loss = self.train_one_epoch(dataloader, epoch)

                # if epoch % 5 == 0: # eval every 2 epochs

                val_loss = self.eval_one_epoch(eval_dataloader, epoch, best_v_loss)

                if val_loss < best_v_loss:
                    best_v_loss = val_loss

                # if epoch % 2 == 0: # save every 5 epochs
                self.save_model(epoch=epoch+1, loss=train_loss)

                print("Train loss: %f\nVal loss: %f\n" % (train_loss, val_loss))

            # self.scheduler.step(epoch+1, loss/len(dataloader))
        
        np.savetxt(os.path.join(self.save_path, "train_loss.csv"), 
           train_loss_list,
           delimiter =", ", 
           fmt ='% s')
        
        np.savetxt(os.path.join(self.save_path, "val_loss.csv"), 
           val_loss_list,
           delimiter =", ", 
           fmt ='% s')
        
        print("Training finished")

        self.save_model(epoch=self.total_epochs, loss=train_loss)

    def save_model(self, epoch, loss, name="checkpoint"):
        print("Saving model at epoch %d at directory %s" % (epoch, self.save_path))

        torch.save({
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(self.save_path, '%s_epochs=%d.pt'%(name, epoch)))


if __name__ == "__main__":

    noise_model = NoiseModel(max_vehicles=MAX_VEHICLES, batch_size=1, experiment=NAME, num_epochs=100, is_graph=IS_GRAPH)
    noise_model.train_model()
