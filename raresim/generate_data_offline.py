import os, glob 
import tqdm 
import numpy as np

np.set_printoptions(precision=15)

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import NGSimDataset

CSV_PATHS_TRAIN = ["./data/train_i80.csv"]
             
CSV_PATHS_TEST = ["./data/test_i80.csv"]

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

savedir = "./data/pt_unshuffled_mydata"

os.makedirs(os.path.join(savedir, "train"), exist_ok=True)
os.makedirs(os.path.join(savedir, "test"), exist_ok=True)

MAX_VEHICLES = 200
GRAPH=True

dataset_train = NGSimDataset(CSV_PATHS_TRAIN, device=DEVICE, max_vehicles=MAX_VEHICLES, num_envs=1, is_graph=GRAPH)

dataset_test = NGSimDataset(CSV_PATHS_TEST, device=DEVICE, max_vehicles=MAX_VEHICLES, num_envs=1, is_graph=GRAPH)

dataloader_train = DataLoader(dataset_train, batch_size=1, pin_memory=False, shuffle=False)

dataloader_test = DataLoader(dataset_test, batch_size=1, pin_memory=False, shuffle=False)


with tqdm.tqdm(dataset_train, unit="batch") as tepoch:

    for b_idx, (obs0, obs1, rand_indices, idxs, tdiff) in enumerate(tepoch):
        
        if b_idx == len(dataloader_train) - 2:
            break

        tepoch.set_description(f"Training data generation")

        # obs0 = obs0.squeeze()
        # obs1 = obs1.squeeze()
        # rand_indices = rand_indices.squeeze()
        # idxs = idxs.squeeze()
        # tdiff = tdiff.squeeze()

        data = {'q0': obs0, 'q1': obs1, 'idx_order': rand_indices, 't_idx': idxs, 't_diff': tdiff}

        # np.save(os.path.join("./data/npy_unshuffled/train", "%d.npy"%(b_idx)), data)
        torch.save(data, os.path.join(savedir, "train", "%d.pt"%(b_idx)))

with tqdm.tqdm(dataset_test, unit="batch") as tepoch:

    for b_idx, (obs0, obs1, rand_indices, idxs, tdiff) in enumerate(tepoch):
        
        if b_idx == len(dataloader_test) - 2:
            break

        tepoch.set_description(f"Testing data generation")

        # obs0 = obs0.squeeze()
        # obs1 = obs1.squeeze()
        # rand_indices = rand_indices.squeeze()
        # idxs = idxs.squeeze()
        # tdiff = tdiff.squeeze()

        data = {'q0': obs0, 'q1': obs1, 'idx_order': rand_indices, 't_idx': idxs, 't_diff': tdiff}

        # np.save(os.path.join("./data/npy_unshuffled/test", "%d.npy"%(b_idx)), data)
        torch.save(data, os.path.join(savedir, "test", "%d.pt"%(b_idx)))


print("data generation finished")

    