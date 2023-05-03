import os, glob
import numpy as np
import pandas as pd
import time 

import torch
from torch.utils.data import Dataset

from ngsim_env.simulation import NGParallelSim

CSV_PATHS = ["./data/trajectories-0400-0415.csv",
            #  "./data/trajectories-0500-0515.csv", 
            #  "./data/trajectories-0515-0530.csv"
             ]

class NGSimDataset(Dataset):
    def __init__(self, csv_path=CSV_PATHS, device='cpu', max_vehicles=300, num_envs=1):
        self.sim = NGParallelSim(csv_path, 0, no_steering=True, 
                                 device=device, delta_time=0.1, 
                                 max_vehicles=max_vehicles, 
                                 num_env=num_envs)

    def __len__(self):
        return len(self.sim.unique_frame_ids)

    def __getitem__(self, idx):

        obs0, obs1_idm, rand_indices, idx = self.sim.get_state_and_next_state(idx, shuffle=True)
        return obs0, obs1_idm, rand_indices, idx
    
if __name__ == "__main__":

    test_dataset = NGSimDataset(device='cuda')

    start = time.time()

    for i in range(1000):
        result = test_dataset.__getitem__(1000)

    total_time = time.time() - start 

    print(total_time / 1000)
