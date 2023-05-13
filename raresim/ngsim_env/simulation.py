import os, sys
from typing import Dict, List

sys.path.append('../')

from envs.traffic._simulation import ParallelTrafficSim
from envs.traffic._idm import IDMLayer
from highway_env.road.lane import AbstractLane, LineType
from highway_env.vehicle.kinematics import Vehicle
from externals.traffic.road.vehicle.micro_vehicle import MicroVehicle
from envs.traffic.diff_highway_env.lane import *
from envs.traffic.diff_highway_env.kinematics import *

from highway_env.road.road import RoadNetwork as hwRoadNetwork
from highway_env.road.regulation import RegulatedRoad as hwRoad

import numpy as np
import torch as th
import pandas as pd
from torch_geometric.data import Data

FEET_TO_METERS_C = 0.30481
# FEET_TO_METERS_C = 1

class NGParallelSim(ParallelTrafficSim):

    def __init__(self, csv_paths, idx, no_steering: bool, device, delta_time=0.1, max_vehicles=300, num_env=1):

        self.num_env = num_env
        self.speed_limit = 29.0576 # 64 mph --> m/s 
        self.no_steering = no_steering
        self.device = device
        self.idx = idx
        self.delta_time = delta_time
        self.max_vehicles = max_vehicles

        # Get info from csv file
        self.csv_path = csv_paths
        self.process_csv_simple()
        # self.process_csv()

        data = self.df.iloc[idx]
        tstep = data["Frame_ID"]

        df = self.df[self.df["Frame_ID"] == tstep]
        # df = df.sample(frac=1).reset_index(drop=True)
        df = df.reset_index(drop=True)
        self.num_vehicle = len(df) 
        self.num_idm_vehicle = self.num_vehicle
        self.num_auto_vehicle = 0 
        self.num_lane = 5 
        
        super().__init__(self.num_env, self.num_auto_vehicle, self.num_idm_vehicle, self.num_lane, self.speed_limit, self.no_steering, self.device)
        
        self.pacecar_env = True

        self.reset(idx) 

    @staticmethod 
    def find_angle_to_x(v):
        ''' finds the angle between vector v and the x axis '''

        a = np.array(v)
        b = np.array([1, 0])
        c = np.dot(a,b) / np.linalg.norm(a) / np.linalg.norm(b)

        theta = np.arccos(np.clip(c, -1, 1))

        return theta 
    
    @staticmethod 
    def find_angle_to_x_vectorized(v):
        ''' same as find_angle_to_x except is for many rows at a time '''

        a = np.array(v).astype(np.float16)
        b = np.tile(np.array([1, 0]), (len(a),1)).astype(np.float16)
        # b = np.array([1,0])
        c = np.dot(a,b.T) / np.linalg.norm(a) / np.linalg.norm(b)

        thetas = np.arccos(np.clip(c, -1, 1))

        return thetas

    @staticmethod
    def rotate_data(v, theta):
        ''' note: theta should be in radians '''

        v = np.array(v)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                              [np.sin(theta), np.cos(theta)]]) 
        
        assert rot_matrix.shape[-1] == v.shape[0], "rotation matrix and input dimension mismatch: %d %d" % (rot_matrix.shape[-1], v.shape[0])

        return rot_matrix @ v 
    
    @staticmethod 
    def rotate_data_vectorized(v, theta):
        ''' same as rotate_data except is for many rows at a time '''

        v[:,0] = v[:,0] * np.cos(theta) - v[:,1] * np.sin(theta) 
        v[:,1] = v[:,0] * np.sin(theta) + v[:,1] * np.cos(theta)
        
        return v 
    
    def process_csv_simple(self):

        print("loading ngsim dataset from csvs: %s" % (self.csv_path))

        df_list = []
        for csv in self.csv_path:
            df = pd.read_csv(csv, header=0, sep=',')
            df_list.append(df)
        
        df = pd.concat(df_list)

        df = df.sort_values(["Frame_ID", "Lane_ID", "Local_X"], ascending=True)

        self.df = df
        column_values = df[['Frame_ID']].values
        self.unique_frame_ids = np.unique(column_values)

        print("data loading finished. here's a preview: ")
        print(self.df.head(3))

    def process_csv(self):

        print("loading ngsim dataset from csvs: %s" % (self.csv_path))

        df_list = []
        frame_id_counter = 0

        for csv in self.csv_path:
            df = pd.read_csv(csv, header=0, sep=',')
            df["Frame_ID"] = df["Frame_ID"] + frame_id_counter
            frame_id_counter = df["Frame_ID"].max()
            df_list.append(df)
        
        df = pd.concat(df_list)

        df = df[df.v_Class != 1]
        df = df[df.Lane_ID <= 5]

        # df = df.drop(columns=["Global_Time", "v_Width", "Global_X", "Global_Y", "v_Class"])

        '''Columns: Index(['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Local_X', 'Local_Y',
            'v_Length', 'v_Vel', 'v_Acc', 'Lane_ID', 'Preceding', 'Following',
            'Space_Headway', 'Time_Headway'],
            dtype='object')
        '''

        total_time = df["Global_Time"].max() - df["Global_Time"].min()
        total_time *= 0.001

        print("dataset total time: ", total_time)

        df = df[["Lane_ID", "Frame_ID", "Local_X", "Local_Y", "v_Vel", "v_Length",]]

        df["Local_X"] = df["Local_X"] * FEET_TO_METERS_C # convert to meters
        df["Local_Y"] = df["Local_Y"] * FEET_TO_METERS_C # convert to meters
        df["v_Vel"] = df["v_Vel"] * FEET_TO_METERS_C # convert to meters
        df["v_Length"] = df["v_Length"] * FEET_TO_METERS_C # convert to meters

        df = df.sort_values(["Frame_ID", "Lane_ID", "Local_X"], ascending=True)
    
        df['theta_to_x'] = -1.5708 # the rotation in radians to align NGSim data with x axis 
        v = NGParallelSim.rotate_data_vectorized(df[['Local_X', 'Local_Y']].to_numpy(), df['theta_to_x'].to_numpy())
        df['sim_position_x'], df['sim_position_y'] = v[:,0], v[:,1]
        df['sim_position_x'] = df['sim_position_x'] - df['sim_position_x'].min() # make sure min value is 0
        df['sim_position_y'] = df['sim_position_y'] - df['sim_position_y'].min() # make sure min value is 0
        
        df = df.drop(columns=["theta_to_x"])

        self.df = df
        column_values = df[['Frame_ID']].values
        self.unique_frame_ids = np.unique(column_values)

        train_df = df.sample(frac=0.8,random_state=200)
        test_df = df.drop(train_df.index)

        train_df.to_csv("data/train_i80_400.csv", encoding='utf-8', index=False)
        test_df.to_csv("data/test_i80_400.csv", encoding='utf-8', index=False)

        print("data loading finished. here's a preview: ")
        print(self.df.head(3))

    def rotate_trajectory(self, theta):

        self.df['theta_to_x'] = theta # the rotation in radians to align NGSim data with x axis 
        v = NGParallelSim.rotate_data_vectorized(self.df[['Local_X', 'Local_Y']].to_numpy(), 
                                                                         self.df['theta_to_x'].to_numpy())
        self.df['sim_position_x'], self.df['sim_position_y'] = v[:,0], v[:,1]

        self.df['sim_position_x'] = self.df['sim_position_x'] - self.df['sim_position_x'].min() # make sure min value is 0
        self.df['sim_position_y'] = self.df['sim_position_y'] - self.df['sim_position_y'].min() # make sure min value is 0

        self.df = self.df.drop(columns=["theta_to_x"])

    def revert_trajectory(self):
        ''' reverts sim_position back to original trajectory '''
        self.rotate_trajectory(self, -1.5708)

    def _set_simulation_state_from_df(self, tstep, env_id=[]):
        ''' Converts dataframe to Pytorch Tensor observation format.
            Updates world position, world velocity, and length variables of instance.
            Dataframe must contain rows ["Lane_ID", "Local_X", "Local_Y", "v_Vel", "v_Length"]
        '''
        df = self.df[self.df["Frame_ID"] == tstep]
        # df = df.sample(frac=1).reset_index(drop=True)

        f = lambda x,y: np.sqrt(x**2 + y**2)

        df = df.reset_index(drop=True)
        df["temp"] = np.sqrt(df["sim_position_x"] ** 2 + df["sim_position_y"] ** 2)
        df = df.sort_values(['Lane_ID', 'temp']).drop('temp', axis=1) # order the same as the simulator

        # min_v_id = self.df['Lane_ID'].min()

        # Reshape df["Local_X", "Local_Y"] to self.vehicle_world_position --> (self.num_env, self.num_vehicle, 2)

        self.num_vehicle = len(df) 
        self.num_idm_vehicle = self.num_vehicle
        self.num_auto_vehicle = 0 
        self.num_lane = 5 

        self.reset_vehicle_info()
        
        # make idm vehicles;
        idm_vehicle_id = 0
        lane_max_pos = []
        
        for _, row in df.iterrows():

            nv = MicroVehicle.default_micro_vehicle(self.speed_limit) 

            lane_id = row["Lane_ID"] - 1
            v_length = row["v_Length"]
            position = np.sqrt(row["sim_position_x"] ** 2 + row["sim_position_y"] ** 2)
            velocity = row["v_Vel"]
            
            vid = self.idm_vehicle_id(idm_vehicle_id)

            if env_id or env_id == []:
                self.vehicle_position[env_id, vid] = self.tensorize_value(position)
                self.vehicle_speed[env_id, vid] = self.tensorize_value(velocity)
                self.vehicle_accel_max[env_id, vid] = self.tensorize_value(nv.accel_max)
                self.vehicle_accel_pref[env_id, vid] = self.tensorize_value(nv.accel_pref)
                self.vehicle_target_speed[env_id, vid] = self.tensorize_value(nv.target_speed)
                self.vehicle_min_space[env_id, vid] = self.tensorize_value(nv.min_space)
                self.vehicle_time_pref[env_id, vid] = self.tensorize_value(nv.time_pref)
                self.vehicle_lane_id[env_id, vid] = self.tensorize_value(lane_id, dtype=th.int32)
                self.vehicle_length[env_id, vid] = self.tensorize_value(v_length)
            else:
                self.vehicle_position[:, vid] = self.tensorize_value(position)
                self.vehicle_speed[:, vid] = self.tensorize_value(velocity)
                self.vehicle_accel_max[:, vid] = self.tensorize_value(nv.accel_max)
                self.vehicle_accel_pref[:, vid] = self.tensorize_value(nv.accel_pref)
                self.vehicle_target_speed[:, vid] = self.tensorize_value(nv.target_speed)
                self.vehicle_min_space[:, vid] = self.tensorize_value(nv.min_space)
                self.vehicle_time_pref[:, vid] = self.tensorize_value(nv.time_pref)
                self.vehicle_lane_id[:, vid] = self.tensorize_value(lane_id, dtype=th.int32)
                self.vehicle_length[:, vid] = self.tensorize_value(v_length)

            curr_lane_max_pos = position

            idm_vehicle_id += 1

            lane_max_pos.append(curr_lane_max_pos)

        self.update_info()

    def set_state(self, idx, env_id=[], next_t=False):
        ''' Load simulation state based on row in dataframe. '''
        
        self.idx = idx 

        # data = self.df.iloc[idx]
        # tstep = int(data["Frame_ID"])
        
        tstep0 = tstep = self.unique_frame_ids[idx]
        tdiff = 1 if next_t else 0

        if next_t: 
            idx += 1
            tstep1 = tstep = self.unique_frame_ids[idx]
            tdiff = tstep1 - tstep0
        
        # tstep = self.unique_frame_ids[idx]

        self._set_simulation_state_from_df(tstep, env_id=env_id)

        return tdiff

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
        
    
    def forward(self, noise: th.Tensor = None):
        '''
        Take a forward step for every environment.
        '''

        # 1. update delta values;
        self.update_delta()

        # 2. call IDM function;
        accel_max = self.vehicle_accel_max[:, self.num_auto_vehicle:].clone()
        accel_pref = self.vehicle_accel_pref[:, self.num_auto_vehicle:].clone()
        speed = self.vehicle_speed[:, self.num_auto_vehicle:].clone()
        target_speed = self.vehicle_target_speed[:, self.num_auto_vehicle:].clone()
        pos_delta = self.vehicle_pos_delta[:, self.num_auto_vehicle:].clone()
        speed_delta = self.vehicle_speed_delta[:, self.num_auto_vehicle:].clone()
        min_space = self.vehicle_min_space[:, self.num_auto_vehicle:].clone()
        time_pref = self.vehicle_time_pref[:, self.num_auto_vehicle:].clone()

        acc = IDMLayer.apply(accel_max, accel_pref, speed, target_speed, pos_delta, speed_delta, min_space, time_pref, self.delta_time)

        if noise is not None:

            min_size = min(acc.shape[-1], noise.shape[-1])
            acc = acc.clone()[:min_size] + noise[:min_size]

        # 3. update pos and vel of vehicles;
        self.vehicle_position[:, self.num_auto_vehicle:] = (self.vehicle_position.clone() + self.vehicle_speed.clone() * self.delta_time)[:, self.num_auto_vehicle:]
        self.vehicle_speed[:, self.num_auto_vehicle:] = (self.vehicle_speed[:, self.num_auto_vehicle:].clone() + acc * self.delta_time)

        # 6. move idm vehicles from lane to lane;
        self.update_idm_vehicle_lane_membership()

        self.update_info()

    def get_state_and_next_state(self, idx, shuffle=False, graph=False):
        ''' for noise model training purposes '''

        rand_indices = th.arange(0, self.num_vehicle)
        # theta = None 

        # generate random indices 
        if shuffle:
            rand_indices = th.randperm(self.num_vehicle)

        if graph: 
            assert shuffle == False, ""

        # if random_rotate:
        #     theta = th.rand(1).item() * 2 * th.pi
        #     self.rotate_trajectory(theta)

        # get actual next timestep
        tdiff = self.set_state(idx, env_id=None, next_t=True)

        # record actual next timestep
        # obs_t1 = self.getObservationGraph() if graph else self.getObservation(shuffle_order=rand_indices)
        obs_t1 = self.getObservation(shuffle_order=rand_indices)

        # set initial state
        self.set_state(idx, env_id=None)
        
        # record initial state
        obs_t0 = self.getObservationGraph() if graph else self.getObservation(shuffle_order=rand_indices)

        # if random_rotate: # put the trajectory back lol
        #     self.revert_trajectory()

        rand_indices = th.nn.functional.pad(input=rand_indices, 
                                            pad=(0, self.max_vehicles-rand_indices.shape[-1]), 
                                            mode='constant', value=0)


        return obs_t0, obs_t1, rand_indices, idx, tdiff

    def add_noise_and_forward(self, acc_noises, idxs, rand_indices=None, tdiff=None):
        ''' for noise model training purposes.
            should be called after get_state_and_next_state. '''

        tensors = [] 

        if tdiff is None: 
            tdiff = th.ones_like(acc_noises)

        # iterate through each batch
        for i in range(0, len(acc_noises)):

            # set initial state 
            self.set_state(idxs[i].item(), env_id=None)
            
            # progress idm for certain # tsteps
            for _ in range(0, tdiff[i].item()):

                # forward idm step to next state
                self.forward(noise=acc_noises[i])

                # record next state
                obs_t1_hat = self.getObservation(shuffle_order=rand_indices[i])

            tensors.append(obs_t1_hat.unsqueeze(0))

        obs_t1_hats = th.vstack(tensors)

        return obs_t1_hats

    def reset_vehicle_info(self):

        # vehicle info;
        self.vehicle_position = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_speed = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_accel_max = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_accel_pref = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_target_speed = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_time_pref = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_min_space = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_length = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_lane_id = th.zeros((self.num_env, self.num_vehicle), dtype=th.int32, device=self.device)

        self.vehicle_pos_delta = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_speed_delta = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)

        self.vehicle_world_position = th.zeros((self.num_env, self.num_vehicle, 2), dtype=th.float32, device=self.device)
        self.vehicle_world_velocity = th.zeros((self.num_env, self.num_vehicle, 2), dtype=th.float32, device=self.device)
        self.vehicle_world_heading = th.zeros((self.num_env, self.num_vehicle,), dtype=th.float32, device=self.device)

        self.vehicle_pairwise_distance = th.zeros((self.num_env, self.num_vehicle, self.num_vehicle), dtype=th.float32, device=self.device)

        self.auto_vehicle_past_headway_thresh = th.zeros((self.num_env, self.num_auto_vehicle), dtype=th.float32, device=self.device)

    def reset(self, idx=0):
        
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

        # create vehicles;

        self.set_state(idx, env_id=None)
        self.update_info()

    def reset_env(self, env_id: List[int], idx=0):
        super().reset_env(env_id)

        with th.no_grad():
            self.set_state(idx, env_id=env_id)
            self.update_info()