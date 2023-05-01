import os, glob
import numpy as np
import pandas as pd
import time 

import torch
from torch.utils.data import Dataset

from ngsim_env.onestep_simulation import NGOneStepSim

CSV_PATH = "./data/trajectories-0400-0415.csv"

class NGSimDataset(Dataset):
    def __init__(self, csv_path=CSV_PATH, device='cpu'):
        self.sim = NGOneStepSim(csv_path, no_steering=True, device=device)

    def __len__(self):
        return len(self.sim.df)

    def __getitem__(self, idx):
        self.sim.forward(idx, 0.01)
        return self.sim.getObservation()

if __name__ == "__main__":

    test_dataset = NGSimDataset(device='cuda')

    start = time.time()

    for i in range(1000):
        result = test_dataset.__getitem__(1000)

    total_time = time.time() - start 

    print(total_time / 1000)
