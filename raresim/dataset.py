import os, glob
import numpy as np
import pandas as pd
import time 

import torch
from torch.utils.data import Dataset

from ngsim_env.simulation import NGParallelSim

from torch_geometric.data import Data, Dataset as GeometricDataset, InMemoryDataset
from torch_geometric.data import download_url

CSV_PATHS = ["./data/trajectories-0400-0415.csv",
            #  "./data/trajectories-0500-0515.csv", 
            #  "./data/trajectories-0515-0530.csv"
             ]

class NGSimDataset(Dataset):
    def __init__(self, csv_path=CSV_PATHS, device='cpu', max_vehicles=300, num_envs=1, is_graph=False):
        self.sim = NGParallelSim(csv_path, 0, no_steering=True, 
                                 device=device, delta_time=0.1, 
                                 max_vehicles=max_vehicles, 
                                 num_env=num_envs)
        
        self.is_graph = is_graph 

    def __len__(self):
        return len(self.sim.unique_frame_ids)

    def __getitem__(self, idx):

        obs0, obs1_idm, rand_indices, idx, tdiff = self.sim.get_state_and_next_state(idx, shuffle=False, graph=self.is_graph)
        return obs0, obs1_idm, rand_indices, idx, tdiff
    
class NGSimDatasetOffline(Dataset):
    def __init__(self, datadir, csv_path=CSV_PATHS, device='cpu', max_vehicles=300, num_envs=1, is_graph=False):
        
        self.datadir = datadir

        self.sim = NGParallelSim(csv_path, 0, no_steering=True, 
                                 device=device, delta_time=0.1, 
                                 max_vehicles=max_vehicles, 
                                 num_env=num_envs)
        
        self.data_list = glob.glob(os.path.join(datadir, "*.pt")) if is_graph else glob.glob(os.path.join(datadir, "*.npy")) 
        self.is_graph = is_graph

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        if self.is_graph: 
            data = torch.load(self.data_list[idx])
            obs0 = (data['q0'].x, data['q0'].edge_index)
            obs1 = data['q1']
            rand_indices = data['idx_order']
            idx = torch.tensor(data['t_idx'], dtype=torch.int64)
            tdiff = torch.tensor(data['t_diff'], dtype=torch.int64)

        else:
            data = np.load(self.data_list[idx], allow_pickle=True)

            obs0 = data.item().get('q0')
            obs1 = data.item().get('q1')
            rand_indices = data.item().get('idx_order')
            idx = data.item().get('t_idx')
            tdiff = data.item().get('t_diff')

        return obs0, obs1, rand_indices, idx, tdiff
    
class NGSimDatasetOfflineGraph(GeometricDataset):
    def __init__(self, root, csv_path=CSV_PATHS, device='cpu', max_vehicles=300, num_envs=1, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data_list = glob.glob(os.path.join(self.root, "*.pt"))

        self.sim = NGParallelSim(csv_path, 0, no_steering=True, 
                                 device=device, delta_time=0.1, 
                                 max_vehicles=max_vehicles, 
                                 num_env=num_envs)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return glob.glob(os.path.join(self.root, "*.pt"))

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        obs0 = data['q0']
        obs1 = data['q1']
        rand_indices = data['idx_order']
        idx = torch.tensor(data['t_idx'], dtype=torch.int64)
        tdiff = torch.tensor(data['t_diff'], dtype=torch.int64)

        return obs0, obs1, rand_indices, idx, tdiff

class ForwardSimGraphDataset(InMemoryDataset):
    def __init__(self, root, x, device='cpu', max_vehicles=300, num_envs=1, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.x = x

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return 1

    def get(self, idx):
        return self.x

if __name__ == "__main__":

    test_dataset = NGSimDataset(device='cuda')

    start = time.time()

    for i in range(1000):
        result = test_dataset.__getitem__(1000)

    total_time = time.time() - start 

    print(total_time / 1000)
