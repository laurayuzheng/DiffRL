import os, glob 
import numpy as np 
import pandas as pd

import torch 
from torch.utils.data import Dataset

from ngsim_env.simulation import NGParallelSim

CSV_PATH = "./data/trajectories-0400-0415.csv"

class NGSimDataset(Dataset):
    def __init__(self, csv_path=CSV_PATH):
        self.sim = NGParallelSim(csv_path, no_steering=True, device='cpu')

    def __len__(self):
        pass 

    def __getitem__(self, idx):
        pass

if __name__ == "__main__":

    test_dataset = NGSimDataset()

