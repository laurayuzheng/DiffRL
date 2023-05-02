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

FEET_TO_METERS_C = 0.30481
# FEET_TO_METERS_C = 1

class NGParallelSim(ParallelTrafficSim):

    def __init__(self, csv_path, idx, no_steering: bool, device, delta_time=0.1):

        self.num_env = 1 
        self.speed_limit = 29.0576 # 64 mph --> m/s 
        self.no_steering = no_steering
        self.device = device
        self.idx = idx
        self.delta_time = delta_time

        # Get info from csv file
        self.csv_path = csv_path
        self.process_csv()

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
    

    def process_csv(self):

        print("loading ngsim dataset from csv: %s" % (self.csv_path))

        df = pd.read_csv(self.csv_path, header=0, sep=',')
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

        df = df.reset_index(drop=True)
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

        data = self.df.iloc[idx]
        tstep = int(data["Frame_ID"])

        if next_t: 
            tstep += 1

        self._set_simulation_state_from_df(tstep, env_id=env_id)

    def getObservation(self, shuffle_order=None):

        position_x = self.vehicle_world_position[:, :, 0]
        position_y = self.vehicle_world_position[:, :, 1]
        velocity_x = self.vehicle_world_velocity[:, :, 0]
        velocity_y = self.vehicle_world_velocity[:, :, 1]
        accel_max = self.vehicle_accel_max[:, :]
        accel_pref = self.vehicle_accel_pref[:, :]
        target_speed = self.vehicle_target_speed[:, :]
        min_space = self.vehicle_min_space[:, :]
        time_pref = self.vehicle_time_pref[:, :]
        vehicle_length = self.vehicle_length[:, :]

        data = [position_x, position_y, velocity_x, velocity_y, accel_max,
                accel_pref, target_speed, min_space, time_pref, vehicle_length]

        if shuffle_order is not None: 

            for i, d in enumerate(data):
                data[i] = d[:, shuffle_order]

        obs_buf = th.cat(data, dim=1)
        
        return obs_buf.squeeze()
    
    def forward(self, actions: th.Tensor, delta_time: float = None):
        super().forward(actions=actions, delta_time=self.delta_time)

    def set_state_and_forward(self, idx, shuffle=False, random_rotate=False):
        ''' for noise model training purposes '''

        rand_indices = None
        theta = None 

        # generate random indices 
        if shuffle:
            rand_indices = th.randperm(self.num_vehicle)

        if random_rotate:
            theta = th.rand(1).item() * 2 * th.pi
            self.rotate_trajectory(theta)

        # get actual next timestep
        self.set_state(idx, env_id=None, next_t=True)

        # record actual next timestep
        obs_t1 = self.getObservation(shuffle_order=rand_indices)

        # set initial state
        self.set_state(idx, env_id=None)
        
        # record initial state
        obs_t0 = self.getObservation(shuffle_order=rand_indices)

        # forward idm step to next state
        self.forward(None)

        # record next state
        obs_t1_hat = self.getObservation(shuffle_order=rand_indices)

        if random_rotate: # put the trajectory back lol
            self.revert_trajectory()

        return obs_t0, obs_t1_hat, obs_t1

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

        self.set_state(self.idx, env_id=None)
        self.update_info()

    def reset_env(self, env_id: List[int], idx=0):
        super().reset_env(env_id)

        with th.no_grad():
            self.set_state(self.idx, env_id=env_id)
            self.update_info()