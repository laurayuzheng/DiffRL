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

FEET_TO_METERS_C = 0.3048

class NGOneStepSim(ParallelTrafficSim):

    def __init__(self, csv_path, no_steering: bool, device):

        # super().__init__(num_env, num_auto_vehicle, num_idm_vehicle, num_lane, speed_limit, no_steering, device)

        self.pacecar_env = True

        # Get info from csv file
        self.num_env = 1
        self.num_lane = 5
        self.speed_limit = 105
        self.device = device 
        self.no_steering = no_steering
        self.csv_path = csv_path

        self.pacecar_env = True
        self.auto_vehicle_max_headway = 8.
        self.emergency_braking_accel = -10.

        self.process_csv()

        df = self.get_data(0)
        self.df_to_observation(df)

        self.reset() 

    def reset(self):

        self.hw_network = hwRoadNetwork()
        self.hw_road = hwRoad(self.hw_network, None)
        self.hw_lane: Dict[int, AbstractLane] = {}

        self.next_lane: Dict[int, List[int]] = {}
        self.next_lane_tensor = th.zeros((self.num_lane, self.num_lane), dtype=th.int32, device=self.device)
        self.next_lane_connectivity = th.zeros((self.num_lane, self.num_lane), dtype=th.int32, device=self.device)
        self.next_lane_space = th.zeros((self.num_env, self.num_lane,), dtype=th.float32, device=self.device)

        self.lane_length = th.zeros((self.num_lane,), dtype=th.float32, device=self.device)

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

        # straight lane info;
        self.straight_lane_start = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)
        self.straight_lane_end = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)
        self.straight_lane_heading = th.zeros((self.num_lane,), dtype=th.float32, device=self.device)
        self.straight_lane_direction = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)
        self.straight_lane_direction_lateral = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)

        # sine lane info;
        self.sine_lane_start = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)
        self.sine_lane_end = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)
        self.sine_lane_heading = th.zeros((self.num_lane,), dtype=th.float32, device=self.device)
        self.sine_lane_direction = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)
        self.sine_lane_direction_lateral = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)
        self.sine_lane_amplitude = th.zeros((self.num_lane,), dtype=th.float32, device=self.device)
        self.sine_lane_pulsation = th.zeros((self.num_lane,), dtype=th.float32, device=self.device)
        self.sine_lane_phase = th.zeros((self.num_lane,), dtype=th.float32, device=self.device)

        # circular lane info;
        self.circular_lane_center = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)
        self.circular_lane_radius = th.zeros((self.num_lane,), dtype=th.float32, device=self.device)
        self.circular_lane_start_phase = th.zeros((self.num_lane,), dtype=th.float32, device=self.device)
        self.circular_lane_end_phase = th.zeros((self.num_lane,), dtype=th.float32, device=self.device)
        self.circular_lane_clockwise = th.zeros((self.num_lane,), dtype=th.int32, device=self.device)

        self.straight_lane_ids = []
        self.sine_lane_ids = []
        self.circular_lane_ids = []

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


    def process_csv(self):

        print("loading ngsim dataset from csv: %s" % (self.csv_path))

        df = pd.read_csv(self.csv_path, header=0, sep=',')
        df = df[df.v_Class != 1]
        # df = df.drop(columns=["Global_Time", "v_Width", "Global_X", "Global_Y", "v_Class"])

        '''Columns: Index(['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Local_X', 'Local_Y',
            'v_Length', 'v_Vel', 'v_Acc', 'Lane_ID', 'Preceding', 'Following',
            'Space_Headway', 'Time_Headway'],
            dtype='object')
        '''

        df = df[["Lane_ID", "Frame_ID", "Local_X", "Local_Y", "v_Vel", "v_Length"]]

        df["Local_X"] = df["Local_X"] * FEET_TO_METERS_C # convert to meters
        df["Local_Y"] = df["Local_Y"] * FEET_TO_METERS_C # convert to meters
        df["v_Vel"] = df["v_Vel"] * FEET_TO_METERS_C # convert to meters
        df["v_Length"] = df["v_Length"] * FEET_TO_METERS_C # convert to meters

        df = df.sort_values(["Frame_ID", "Lane_ID", "Local_X"], ascending=True)

        self.df = df

        print("data loading finished. here's a preview: ")
        print(self.df.head(5))

    def _load_state_at_frame(self, t: int):
        ''' Helper function. Load the simulation state based on time frame t.  '''

        # Filter for all rows belonging to timestep
        sim_state_t = self.df["Frame_ID" == t]

    def df_to_observation(self, df):
        ''' Converts dataframe to Pytorch Tensor observation format.
            Updates world position, world velocity, and length variables of instance.
            Dataframe must contain rows ["Lane_ID", "Local_X", "Local_Y", "v_Vel", "v_Length"]
        '''
        
        df = df.sample(frac=1).reset_index(drop=True)
        # df = df.reset_index(drop=True)
        min_v_id = df['Lane_ID'].min()

        # Reshape df["Local_X", "Local_Y"] to self.vehicle_world_position --> (self.num_env, self.num_vehicle, 2)

        self.num_vehicle = len(df) 
        self.num_idm_vehicle = self.num_vehicle
        self.num_auto_vehicle = 0 
        self.num_lane = 5 
        nv = MicroVehicle.default_micro_vehicle(self.speed_limit) 

        vehicle_world_position = th.zeros((self.num_env, self.num_vehicle, 2), dtype=th.float32, device=self.device)
        vehicle_world_velocity = th.zeros((self.num_env, self.num_vehicle, 2), dtype=th.float32, device=self.device)
        vehicle_position = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        vehicle_speed = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        vehicle_length = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        vehicle_lane_id = th.zeros((self.num_env, self.num_vehicle), dtype=th.int32, device=self.device)
        self.vehicle_accel_max = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_accel_pref = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_target_speed = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_time_pref = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        self.vehicle_min_space = th.zeros((self.num_env, self.num_vehicle), dtype=th.float32, device=self.device)
        
        for index, row in df.iterrows():
            vehicle_world_position[0, index, 0] = row["Local_X"]
            vehicle_world_position[0, index, 1] = row["Local_Y"]
            vehicle_world_velocity[0, index, 0] = row["v_Vel"]
            vehicle_position[0, index] = row["Local_X"]
            vehicle_speed[0, index] = row["v_Vel"]
            vehicle_length[0, index] = row["v_Length"]
            vehicle_lane_id[0, index] = row["Lane_ID"] - min_v_id # min-index --> 0-index
            self.vehicle_accel_max[0, index] = self.tensorize_value(nv.accel_max)
            self.vehicle_accel_pref[0, index] = self.tensorize_value(nv.accel_pref)
            self.vehicle_target_speed[0, index] = self.tensorize_value(nv.target_speed)
            self.vehicle_min_space[0, index] = self.tensorize_value(nv.min_space)
            self.vehicle_time_pref[0, index] = self.tensorize_value(nv.time_pref)

        # Set instance values to new tensors
        self.vehicle_world_position = vehicle_world_position
        self.vehicle_world_velocity = vehicle_world_velocity
        self.vehicle_length = vehicle_length
        self.vehicle_position = vehicle_position
        self.vehicle_speed = vehicle_speed
        self.vehicle_lane_id = vehicle_lane_id

    def get_data(self, idx):
        ''' Load simulation state based on row in dataframe. '''

        data = self.df.iloc[idx]
        tstep = data["Frame_ID"]

        return self.df[self.df["Frame_ID"] == tstep]

    def update_delta(self):

        '''
        Update [pos_delta] and [vel_delta].
        '''

        INF = 1e7

        vehicle_position = self.vehicle_position.clone()
        vehicle_lane_id = self.vehicle_lane_id.clone()
        vehicle_speed = self.vehicle_speed.clone()
        vehicle_length = self.vehicle_length.clone()
        next_lane_space = self.next_lane_space.clone()

        vehicle_lane_id_a = vehicle_lane_id.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self.num_env, self.num_vehicle))
        vehicle_lane_id_b = vehicle_lane_id.unsqueeze(0).unsqueeze(0).expand((self.num_env, self.num_vehicle, -1, -1))

        # vehicle_lane_id_diff[a, b, c, d] = (a'th env's b'th vehicle's lane id) - (c'th env's d'th vehicle's lane id)
        vehicle_lane_id_diff = vehicle_lane_id_a - vehicle_lane_id_b
        # vehicle_lane_id_diff[a, b, c] = (a'th env's c'th vehicle's lane id) - (a'th env's b'th vehicle's lane id)
        vehicle_lane_id_diff = th.diagonal(vehicle_lane_id_diff, dim1=0, dim2=2).transpose(0, 2)

        # ==========================

        vehicle_position_a = vehicle_position.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self.num_env, self.num_vehicle))
        vehicle_position_b = vehicle_position.unsqueeze(0).unsqueeze(0).expand((self.num_env, self.num_vehicle, -1, -1))

        # vehicle_position_diff[a, b, c, d] = (a'th env's b'th vehicle's position) - (c'th env's d'th vehicle's position)
        vehicle_position_diff = vehicle_position_a - vehicle_position_b
        # vehicle_position_diff[a, b, c] = (a'th env's c'th vehicle's position) - (a'th env's b'th vehicle's position)
        vehicle_position_diff = th.diagonal(vehicle_position_diff, dim1=0, dim2=2).transpose(0, 2)

        # do not care about negative position diffs;
        tmp_idx = th.where(vehicle_position_diff < 0.)
        vehicle_position_diff[tmp_idx] = INF
        # do not care about different lanes;
        tmp_idx = th.where(vehicle_lane_id_diff != 0)
        vehicle_position_diff[tmp_idx] = INF
        # do not care about same vehicles;
        tmp_add = th.eye(self.num_vehicle, dtype=th.float32, device=self.device).unsqueeze(0) * INF
        vehicle_position_diff = vehicle_position_diff + tmp_add

        vehicle_pos_delta, leading_vehicle_id = vehicle_position_diff.min(dim=2)
        leading_vehicle_speed = th.gather(vehicle_speed, 1, leading_vehicle_id)
        leading_vehicle_length = th.gather(vehicle_length, 1, leading_vehicle_id)

        vehicle_pos_delta = vehicle_pos_delta - (vehicle_length + leading_vehicle_length) * 0.5
        vehicle_vel_delta = vehicle_speed - leading_vehicle_speed

        expanded_lane_length = self.lane_length.unsqueeze(0).expand((self.num_env, -1))

        vehicle_lane_length = th.gather(expanded_lane_length, 1, vehicle_lane_id.to(dtype=th.int64))
        expanded_next_lane_space = next_lane_space.unsqueeze(1).expand((self.num_env, self.num_vehicle, self.num_lane))
        vehicle_next_lane_space = th.gather(expanded_next_lane_space, 2, vehicle_lane_id.unsqueeze(-1).to(dtype=th.int64)).squeeze(-1)
        vehicle_max_pos_delta = vehicle_lane_length - vehicle_position - vehicle_length * 0.5 + vehicle_next_lane_space

        self.vehicle_pos_delta = vehicle_pos_delta
        self.vehicle_speed_delta = vehicle_vel_delta

        tmp_idx = th.where(vehicle_pos_delta > vehicle_max_pos_delta)
        self.vehicle_pos_delta[tmp_idx] = vehicle_max_pos_delta[tmp_idx]
        self.vehicle_pos_delta = th.clip(self.vehicle_pos_delta, min=1e-3)
        self.vehicle_speed_delta[tmp_idx] = 0.

        self.auto_vehicle_past_headway_thresh = th.where(self.vehicle_pos_delta[:, :self.num_auto_vehicle] < self.auto_vehicle_max_headway, 1, 0)

    def forward(self, idx, delta_time: float):
        
        df = self.get_data(idx)
        self.df_to_observation(df)

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

        acc = IDMLayer.apply(accel_max, accel_pref, speed, target_speed, pos_delta, speed_delta, min_space, time_pref, delta_time)

        # 3. update pos and vel of vehicles;
        self.vehicle_position[:, self.num_auto_vehicle:] = (self.vehicle_position.clone() + self.vehicle_speed.clone() * delta_time)[:, self.num_auto_vehicle:]
        self.vehicle_speed[:, self.num_auto_vehicle:] = (self.vehicle_speed[:, self.num_auto_vehicle:].clone() + acc * delta_time)

    def getObservation(self):

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

        obs_buf = th.cat([position_x, position_y, velocity_x, velocity_y, accel_max,
                                    accel_pref, target_speed, min_space, time_pref, vehicle_length], dim=1)

        
        return obs_buf.squeeze()