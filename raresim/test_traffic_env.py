# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import time

import torch
import random

torch.set_printoptions(sci_mode=False)

import envs
from utils.common import *

import argparse

from ngsim_env.env import TrafficNGSimEnv
from ngsim_env.simulation import NGParallelSim
from simple_env.simulation import WorldSim
from simple_env.env import TrafficNoiseEnv
from envs.traffic.pace_car.env import TrafficPaceCarEnv


CSV_PATHS = ["./data/train_i80.csv",
            #  "./data/trajectories-0500-0515.csv", 
            #  "./data/trajectories-0515-0530.csv"
             ]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

'''python examples/test_traffic_env.py --num-envs 1 --render'''

parser = argparse.ArgumentParser()
# parser.add_argument('--env', type = str, default = 'TrafficRoundaboutEnv')
parser.add_argument('--num-envs', type = int, default = 2)
parser.add_argument('--render', default = False, action = 'store_true')

args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

seeding()

###########################
# Testing the Simulation #
###########################

# start_idx = 5000
# t_start = time.time()

# # sim = NGParallelSim(CSV_PATH, 0, True, device, delta_time=0.1)
# sim = WorldSim(100, 50)

# for i in range(1000):
#     sim.forward()

# t_end = time.time()


###########################
# Testing the Environment #
###########################

env = TrafficNoiseEnv(num_idm_vehicle=200, render=args.render, device=device,
                      num_envs=args.num_envs, seed=0,
                      episode_length=1000, no_grad=True, no_steering=True)
        
# env = TrafficPaceCarEnv(device="cpu", num_envs=100, num_auto_vehicle=0, num_idm_vehicle=50, num_lane=5)

# obs = env.reset(idx=5000)
obs = env.reset()

num_actions = env.num_actions

t_start = time.time()

reward_episode = 0.
for i in range(900):
    actions = torch.zeros((args.num_envs, num_actions), device=device)
    # actions = torch.Tensor([0]).expand(args.num_envs, -1).to(device)
    obs, reward, _, _ = env.step(actions)
    reward_episode += reward

t_end = time.time()

print("Total time: %f s", (t_end - t_start))
print("Steps / second: ", 900 / (t_end - t_start))

# print('fps = ', 1000 * args.num_envs / (t_end - t_start))
# print('mean reward = ', reward_episode.mean().detach().cpu().item())

print('Finish Successfully')

