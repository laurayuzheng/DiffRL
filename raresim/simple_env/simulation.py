import os, sys
from typing import List

sys.path.append("../")

import numpy as np
import torch as th
from torch_geometric.data import Data

from highway_env.road.lane import AbstractLane, LineType
from highway_env.vehicle.kinematics import Vehicle

from externals.traffic.road.vehicle.micro_vehicle import MicroVehicle

from envs.traffic._simulation import ParallelTrafficSim
from envs.traffic._idm import IDMLayer

class WorldSim(ParallelTrafficSim):

    ''' In this simulator, we load a trained noise model and perform realistic forward simulation. 
        Simulations in each env are not restarted. 
        When environments reach a "done" state, the environment remains at the terminal end state until all other envs finish running.
    '''

    def __init__(self, num_env: int, num_idm_vehicle: int, max_vehicles=200, no_steering: bool=True, device="cpu"):

        super().__init__(num_env, 0, num_idm_vehicle, 5, 29.0576, no_steering, device)
        self.pacecar_env = True
        # self.init_position, self.init_speed = init_state # [num_env, 2] and [num_env, 1]
        self.initial_state = None 
        self.delta_time = 0.1
        self.max_vehicles = max_vehicles
    
    def set_state_from_obs(self, obs):
        
        # obs = th.tensor(obs, dtype=th.float32)
        obs = obs[1:].clone() # .requires_grad_(True) #.clone().detach().requires_grad_(True)

        obs = obs.reshape((self.num_env, 10, -1))
        obs = obs[:,:,:self.num_idm_vehicle] # get rid of padding

        position_x = obs[:,0] * 1e6 
        position_y = obs[:,1] * (5 * 5)
        velocity_x = obs[:,2] * (self.speed_limit * 2)
        velocity_y = obs[:,3] * (self.speed_limit * 2)

        self.vehicle_position = position_x
        # self.vehicle_position[:, :, 1] = position_y 
        self.vehicle_speed = velocity_x 
        # self.vehicle_speed[:, :, 1] = velocity_y 

        self.update_info()

    def random_initial_state(self):

        '''
        Get random state of the road network.
        '''

        num_lanes = 5
        num_vehicle = self.num_vehicle 
        vehicle_length = 5
        dtype = th.float32

        speed_limit = th.tensor([self.speed_limit], dtype=dtype)

        position_start = th.arange(0, num_vehicle).unsqueeze(0).expand(self.num_lane, -1)\
                        .unsqueeze(0).expand(self.num_env, -1, -1) * 4.0 * vehicle_length # [num_env, n_vehicle]
        init_position = position_start + th.rand((self.num_env, self.num_lane, num_vehicle), dtype=dtype) * 2.0 * vehicle_length
        
        init_speed = th.rand((self.num_env, self.num_lane, num_vehicle), dtype=dtype)
        init_speed = th.lerp(0.3 * speed_limit, 0.7 * speed_limit, init_speed)

        mask = th.randperm(self.num_env * self.num_lane * num_vehicle).reshape((self.num_env, self.num_lane, num_vehicle))
        mask = th.where(mask >= num_vehicle*self.num_env, 0, 1)

        init_position = init_position * mask 

        assert th.count_nonzero(init_position) == num_vehicle*self.num_env

        init_position = init_position[th.nonzero(init_position, as_tuple=True)].reshape(self.num_env, num_vehicle)
        init_speed = init_speed * mask 

        assert th.count_nonzero(init_speed) == num_vehicle*self.num_env

        init_speed = init_speed[th.nonzero(init_speed, as_tuple=True)].reshape(self.num_env, num_vehicle)

        # lane_ids = th.tile(th.tile(th.arange(0, num_lanes).T, (num_vehicle, 1)), (1, self.num_env))
        lane_ids = th.arange(1, num_lanes+1).reshape(num_lanes, 1)
        lane_ids = th.tile(th.tile(lane_ids, (1, num_vehicle)).unsqueeze(0), (self.num_env, 1, 1))

        lane_ids = lane_ids * mask 

        assert th.count_nonzero(lane_ids) == num_vehicle*self.num_env

        lane_ids = lane_ids[th.nonzero(lane_ids, as_tuple=True)].reshape(self.num_env, num_vehicle)
        lane_ids = lane_ids - 1

        return init_position, init_speed, lane_ids 
    

    def reset(self):

        super().reset()

        # add straight lanes;
        lane_width = AbstractLane.DEFAULT_WIDTH
        for i in range(self.num_lane):
            self.lane_length[i] = 1e6

            start = np.array([0, float(i) * lane_width])
            end = np.array([self.lane_length[i].cpu().item(), float(i) * lane_width])
            
            line_type = [LineType.STRIPED, LineType.STRIPED]

            if i == 0:
                line_type[0] = LineType.CONTINUOUS
            if i == self.num_lane - 1:
                line_type[1] = LineType.CONTINUOUS

            self.make_straight_lane(i, start, end, "start", "end", line_type)

            # go back to itself;
            self.next_lane[i] = [i]

        self.fill_next_lane_tensor()

        self.set_random_state()

    def set_random_state(self):
        init_position, init_speed, lane_ids = self.random_initial_state()
        
        self.init_position = init_position 
        self.init_speed = init_speed 
        self.lane_ids = lane_ids

        nv = MicroVehicle.default_micro_vehicle(self.speed_limit)

        self.vehicle_position = self.tensorize_value(self.init_position.reshape((self.num_env, -1)))
        self.vehicle_speed = self.tensorize_value(self.init_speed.reshape((self.num_env, -1)))
        self.vehicle_accel_max = self.tensorize_value(th.ones_like(self.vehicle_accel_max) * nv.accel_max)
        self.vehicle_accel_pref = self.tensorize_value(th.ones_like(self.vehicle_accel_pref) * nv.accel_pref)
        self.vehicle_target_speed = self.tensorize_value(th.ones_like(self.vehicle_target_speed) * nv.target_speed)
        self.vehicle_min_space = self.tensorize_value(th.ones_like(self.vehicle_min_space) * nv.min_space)
        self.vehicle_time_pref = self.tensorize_value(th.ones_like(self.vehicle_time_pref) * nv.time_pref)
        self.vehicle_lane_id = self.tensorize_value(self.lane_ids, dtype=th.int32)
        self.vehicle_length = self.tensorize_value(th.ones_like(self.vehicle_time_pref) * nv.length)

        self.update_info()

        self.active_envs = th.arange(0, self.num_env)

    def getRandomObservation(self):

        init_position, init_speed, lane_ids = self.random_initial_state()

        vehicle_lane_id = lane_ids.clone()
        vehicle_position = init_position.clone()
        vehicle_speed = init_speed.clone()

        # (num env, num vehicle, num lane)
        lane_id_tensor = th.arange(0, self.num_lane, dtype=th.int32, device=self.device)
        lane_id_tensor = lane_id_tensor.unsqueeze(0).unsqueeze(0).expand((self.num_env, self.num_vehicle, self.num_lane))
        vehicle_lane_id_tensor = vehicle_lane_id.unsqueeze(-1).expand((self.num_env, self.num_vehicle, self.num_lane))
        vehicle_lane_id_tensor = vehicle_lane_id_tensor - lane_id_tensor
        vehicle_lane_membership_tensor = vehicle_lane_id_tensor == 0

        # (num_env, num lane, num vehicle)
        lane_vehicle_membership_tensor = vehicle_lane_membership_tensor.transpose(1, 2)
        lane_vehicle_position_tensor = vehicle_position.unsqueeze(1).expand((self.num_env, self.num_lane, self.num_vehicle))
        inf_tensor = th.zeros((self.num_env, self.num_lane, self.num_vehicle))
        lane_vehicle_position_tensor = th.where(lane_vehicle_membership_tensor,
                                                    lane_vehicle_position_tensor,
                                                    inf_tensor)
        
        lane_vehicle_speed_tensor = vehicle_speed.unsqueeze(1).expand((self.num_env, self.num_lane, self.num_vehicle))
        lane_vehicle_speed_tensor = th.where(lane_vehicle_membership_tensor,
                                                    lane_vehicle_speed_tensor,
                                                    inf_tensor)

        start_nodes = th.nonzero(lane_vehicle_membership_tensor)

        start_nodes = start_nodes[:,1:].transpose(0,1) #.flip(1)

        edges = []
        lane_node_list = [[] for _ in range(self.num_lane)] 

        for i in range(start_nodes.shape[1]):
            lane_node_list[start_nodes[0][i]].append(int(start_nodes[1][i])+1)

        for i in range(len(lane_node_list)):
            for j in range(len(lane_node_list[i])):
                if j == len(lane_node_list[i]) - 1:
                    edges.append([0, lane_node_list[i][j]])
                else:
                    edges.append([lane_node_list[i][j+1], lane_node_list[i][j]])
        
        edge_index = th.tensor(edges, dtype=th.long).transpose(0,1) # edges for graph data done

        nv = MicroVehicle.default_micro_vehicle(self.speed_limit)

        vehicle_position = self.tensorize_value(init_position.reshape((self.num_env, -1)))
        vehicle_speed = self.tensorize_value(init_speed.reshape((self.num_env, -1)))
        vehicle_accel_max = self.tensorize_value(th.ones_like(self.vehicle_accel_max) * nv.accel_max)
        vehicle_accel_pref = self.tensorize_value(th.ones_like(self.vehicle_accel_pref) * nv.accel_pref)
        vehicle_target_speed = self.tensorize_value(th.ones_like(self.vehicle_target_speed) * nv.target_speed)
        vehicle_min_space = self.tensorize_value(th.ones_like(self.vehicle_min_space) * nv.min_space)
        vehicle_time_pref = self.tensorize_value(th.ones_like(self.vehicle_time_pref) * nv.time_pref)
        vehicle_lane_id = self.tensorize_value(lane_ids, dtype=th.int32)
        vehicle_length = self.tensorize_value(th.ones_like(self.vehicle_time_pref) * nv.length)

        vehicle_world_position, vehicle_world_velocity, vehicle_world_heading = self.calculateWorldValues(vehicle_position, vehicle_speed, vehicle_lane_id)

        nv = MicroVehicle.default_micro_vehicle(self.speed_limit) 

        position_x = vehicle_world_position[:, :, 0] / 1e6 # max lane length
        position_y = vehicle_world_position[:, :, 1] / (5 * 5) # width * num lanes
        velocity_x = vehicle_world_velocity[:, :, 0] / (self.speed_limit * 2)
        velocity_y = vehicle_world_velocity[:, :, 1] / (self.speed_limit * 2)
        accel_max = vehicle_accel_max[:, :] / (nv.accel_max*2)
        accel_pref = vehicle_accel_pref[:, :] / (nv.accel_pref*2)
        target_speed = vehicle_target_speed[:, :] / (nv.target_speed*2)
        min_space = vehicle_min_space[:, :] / (nv.min_space*2)
        time_pref = vehicle_time_pref[:, :] / (nv.time_pref*2)
        vehicle_length = vehicle_length[:, :] / (nv.length*2)
        
        data = [position_x, position_y, velocity_x, velocity_y, accel_max,
                accel_pref, target_speed, min_space, time_pref, vehicle_length]

        # zero-pad observation to have a consistent size across all data
        # for i, d in enumerate(data):
        #     data[i] = th.nn.functional.pad(input=d, pad=(0, self.max_vehicles - d.shape[-1], 0, 0), mode='constant', value=0)

        # obs_buf = th.cat(data, dim=1)
        obs_buf = th.cat(data, dim=0).transpose(0,1)
        dummy_root_data = th.ones_like(obs_buf[0])
        obs_buf = th.vstack([dummy_root_data, obs_buf])

        data = Data(x=obs_buf, edge_index=edge_index)

        return data

    def forward(self, noise: th.Tensor = None):
        '''
        Take a forward step for every environment.
        '''

        # 1. update delta values;
        self.update_delta()

        # 2. call IDM function;
        accel_max = self.vehicle_accel_max[self.active_envs, self.num_auto_vehicle:].clone()
        accel_pref = self.vehicle_accel_pref[self.active_envs, self.num_auto_vehicle:].clone()
        speed = self.vehicle_speed[self.active_envs, self.num_auto_vehicle:].clone()
        target_speed = self.vehicle_target_speed[self.active_envs, self.num_auto_vehicle:].clone()
        pos_delta = self.vehicle_pos_delta[self.active_envs, self.num_auto_vehicle:].clone()
        speed_delta = self.vehicle_speed_delta[self.active_envs, self.num_auto_vehicle:].clone()
        min_space = self.vehicle_min_space[self.active_envs, self.num_auto_vehicle:].clone()
        time_pref = self.vehicle_time_pref[self.active_envs, self.num_auto_vehicle:].clone()

        acc = IDMLayer.apply(accel_max, accel_pref, speed, target_speed, pos_delta, speed_delta, min_space, time_pref, self.delta_time)

        if noise is not None:
        
            start_i = noise.shape[1] - acc.shape[1]
            noise = noise[:, start_i:]
            noise = noise.reshape(acc.shape)

            min_size = min(acc.shape[-1], noise.shape[-1])
            acc = acc.clone()[:min_size] + noise[:min_size]

            # clip negative velocities 

            v_next = speed + acc * self.delta_time
            acc = th.where(v_next >= 0, acc, -speed / self.delta_time)

        if self.active_envs.nelement() > 0: # small optimization
            
            # 3. update pos and vel of vehicles;
            self.vehicle_position[self.active_envs, self.num_auto_vehicle:] = (self.vehicle_position.clone() + self.vehicle_speed.clone() * self.delta_time)[self.active_envs, self.num_auto_vehicle:]
            self.vehicle_speed[self.active_envs, self.num_auto_vehicle:] = (self.vehicle_speed.clone() + acc * self.delta_time)[self.active_envs, self.num_auto_vehicle:]

            # 6. move idm vehicles from lane to lane;
            self.update_idm_vehicle_lane_membership()

            self.update_info()

    def reset_env(self, env_id: List[int], restart_all=False):
        super().reset_env(env_id)

        if restart_all:
            self.active_envs = th.arange(0, self.num_env)
        else:
            self.active_envs = self.active_envs[self.active_envs!=self.active_envs[env_id]]

    def getObservationGraph(self):

        # class MyData(Data):
        #     def __cat_dim__(self, key, value, *args, **kwargs):
        #         if key == 'x':
        #             return None
        #         return super().__cat_dim__(key, value, *args, **kwargs)
            
        vehicle_lane_id = self.vehicle_lane_id.clone()
        vehicle_position = self.vehicle_position.clone()
        vehicle_speed = self.vehicle_speed.clone()

        # (num env, num vehicle, num lane)
        lane_id_tensor = th.arange(0, self.num_lane, dtype=th.int32, device=self.device)
        lane_id_tensor = lane_id_tensor.unsqueeze(0).unsqueeze(0).expand((self.num_env, self.num_vehicle, self.num_lane))
        vehicle_lane_id_tensor = vehicle_lane_id.unsqueeze(-1).expand((self.num_env, self.num_vehicle, self.num_lane))
        vehicle_lane_id_tensor = vehicle_lane_id_tensor - lane_id_tensor
        vehicle_lane_membership_tensor = vehicle_lane_id_tensor == 0

        # (num_env, num lane, num vehicle)
        lane_vehicle_membership_tensor = vehicle_lane_membership_tensor.transpose(1, 2)
        lane_vehicle_position_tensor = vehicle_position.unsqueeze(1).expand((self.num_env, self.num_lane, self.num_vehicle))
        # inf_tensor = self.lane_length.unsqueeze(0).unsqueeze(-1).expand((self.num_env, self.num_lane, self.num_vehicle))
        inf_tensor = th.zeros((self.num_env, self.num_lane, self.num_vehicle))
        lane_vehicle_position_tensor = th.where(lane_vehicle_membership_tensor,
                                                    lane_vehicle_position_tensor,
                                                    inf_tensor)
        
        lane_vehicle_speed_tensor = vehicle_speed.unsqueeze(1).expand((self.num_env, self.num_lane, self.num_vehicle))
        lane_vehicle_speed_tensor = th.where(lane_vehicle_membership_tensor,
                                                    lane_vehicle_speed_tensor,
                                                    inf_tensor)

        start_nodes = th.nonzero(lane_vehicle_membership_tensor)

        start_nodes = start_nodes[:,1:].transpose(0,1) #.flip(1)

        edges = []
        lane_node_list = [[] for _ in range(self.num_lane)] 

        for i in range(start_nodes.shape[1]):
            lane_node_list[start_nodes[0][i]].append(int(start_nodes[1][i])+1)

        for i in range(len(lane_node_list)):
            for j in range(len(lane_node_list[i])):
                if j == len(lane_node_list[i]) - 1:
                    edges.append([0, lane_node_list[i][j]])
                else:
                    edges.append([lane_node_list[i][j+1], lane_node_list[i][j]])
        
        edge_index = th.tensor(edges, dtype=th.long).transpose(0,1) # edges for graph data done

        # now build the data tensor
        nv = MicroVehicle.default_micro_vehicle(self.speed_limit) 

        position_x = self.vehicle_world_position[:, :, 0] / 1e6 # max lane length
        position_y = self.vehicle_world_position[:, :, 1] / (5 * 5) # width * num lanes
        velocity_x = self.vehicle_world_velocity[:, :, 0] / (self.speed_limit * 2)
        velocity_y = self.vehicle_world_velocity[:, :, 1] / (self.speed_limit * 2)
        accel_max = self.vehicle_accel_max[:, :] / (nv.accel_max*2)
        accel_pref = self.vehicle_accel_pref[:, :] / (nv.accel_pref*2)
        target_speed = self.vehicle_target_speed[:, :] / (nv.target_speed*2)
        min_space = self.vehicle_min_space[:, :] / (nv.min_space*2)
        time_pref = self.vehicle_time_pref[:, :] / (nv.time_pref*2)
        vehicle_length = self.vehicle_length[:, :] / (nv.length*2)

        data = [position_x, position_y, velocity_x, velocity_y, accel_max,
                accel_pref, target_speed, min_space, time_pref, vehicle_length]
        
        obs_buf = th.cat(data, dim=0).transpose(0,1)
        dummy_root_data = th.ones_like(obs_buf[0])
        obs_buf = th.vstack([dummy_root_data, obs_buf])

        data = Data(x=obs_buf, edge_index=edge_index)

        return data 

    def getObservation(self, shuffle_order=None, graph=False):

        nv = MicroVehicle.default_micro_vehicle(self.speed_limit) 

        position_x = self.vehicle_world_position[:, :, 0] / 1e6 # max lane length
        position_y = self.vehicle_world_position[:, :, 1] / (5 * 5) # width * num lanes
        velocity_x = self.vehicle_world_velocity[:, :, 0] / (self.speed_limit * 2)
        velocity_y = self.vehicle_world_velocity[:, :, 1] / (self.speed_limit * 2)
        accel_max = self.vehicle_accel_max[:, :] / (nv.accel_max*2)
        accel_pref = self.vehicle_accel_pref[:, :] / (nv.accel_pref*2)
        target_speed = self.vehicle_target_speed[:, :] / (nv.target_speed*2)
        min_space = self.vehicle_min_space[:, :] / (nv.min_space*2)
        time_pref = self.vehicle_time_pref[:, :] / (nv.time_pref*2)
        vehicle_length = self.vehicle_length[:, :] / (nv.length*2)
        
        data = [position_x, position_y, velocity_x, velocity_y, accel_max,
                accel_pref, target_speed, min_space, time_pref, vehicle_length]

        if shuffle_order is not None: 
            
            shuffle_order = list(shuffle_order.detach().cpu().numpy())
            shuffle_order_size = len(shuffle_order)
            position_x_size = len(position_x[0])

            if max(shuffle_order) >= position_x_size:
                
                max_index = max(shuffle_order)
                # while shuffle_order_size > position_x_size:
                while max_index >= position_x_size:
                    shuffle_order.remove(max_index)
                    max_index = max(shuffle_order)

            elif shuffle_order_size < position_x_size:
        
                # remove new data from obs since it wasn't present originally
                for i, d in enumerate(data): 
                    data[i] = d[:, :shuffle_order_size]

            for i, d in enumerate(data):
                data[i] = d[:, shuffle_order]

        # zero-pad observation to have a consistent size across all data
        for i, d in enumerate(data):
            data[i] = th.nn.functional.pad(input=d, pad=(0, self.max_vehicles - d.shape[-1], 0, 0), mode='constant', value=0)

        obs_buf = th.cat(data, dim=1)

        return obs_buf.squeeze()