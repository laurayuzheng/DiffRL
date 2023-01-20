from envs.traffic._network import ParallelRoadNetwork
from envs.traffic._idm import IDMLayer

from typing import List
import torch as th

class TrafficSim:

    def __init__(self, env_list: List[ParallelRoadNetwork]):

        self.env_list = env_list

    def num_env(self):

        return len(self.env_list)

    def forward(self, actions: th.Tensor, delta_time: float):

        '''
        Take a forward step for every environment.
        '''

        # 1. collect states across envs;
        states = [[] for _ in range(9)]
        num_vehicle_per_env = []
        acc_num_vehicle_per_env = [0]

        for env in self.env_list:

            curr_state = env.tensorize(delta_time)
            num_vehicle_per_env.append(env.num_vehicle)
            acc_num_vehicle_per_env.append(acc_num_vehicle_per_env[-1] + env.num_vehicle)
            
            for i in range(9):
                states[i].append(curr_state[i])

        for i in range(9):

            states[i] = th.cat(states[i])

        # 2. call IDM function;
        acc = IDMLayer.apply(states[0], 
                                states[1], 
                                states[2], 
                                states[3], 
                                states[4], 
                                states[5], 
                                states[6], 
                                states[7], 
                                states[8])

        # 3. update next pos and vel of vehicles;
        for i, env in enumerate(self.env_list):

            idx_start = acc_num_vehicle_per_env[i]
            idx_end = acc_num_vehicle_per_env[i + 1]
            curr_acc = acc[idx_start:idx_end]

            env.update_next_state(curr_acc, delta_time)

        # 4. callback to resolve special cases (e.g. merge);
        for env in self.env_list:
            env.resolve_next_state()

        # 5. apply next pos and vel to every vehicle;
        for env in self.env_list:
            env.update_state()

        # 6. move vehicles from lane to lane;
        for env in self.env_list:
            env.conversion(delta_time)

        # 7. apply action and update state of auto vehicles;
        for i, env in enumerate(self.env_list):
            curr_actions = actions[i]
            curr_steering = curr_actions[0::2]
            curr_acceleration = curr_actions[1::2]
            env.forward_auto(curr_steering, curr_acceleration, delta_time)

        # 8. update tensor info;
        for env in self.env_list:
            env.update_tensor_info()

    def clear_grad(self):
        for env in self.env_list:
            env.clear_grad()