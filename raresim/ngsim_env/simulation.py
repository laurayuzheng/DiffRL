import os, sys 
from typing import List

sys.path.append('../')

from envs.traffic._simulation import ParallelTrafficSim
from highway_env.road.lane import AbstractLane, LineType
from highway_env.vehicle.kinematics import Vehicle
from externals.traffic.road.vehicle.micro_vehicle import MicroVehicle
import numpy as np
import torch as th
import pandas as pd 

FEET_TO_METERS_C = 0.3048

class NGParallelSim(ParallelTrafficSim):

    def __init__(self, csv_path, no_steering: bool, device):
        
        num_env = 1
        num_auto_vehicle = 0
        num_idm_vehicle = 0 
        num_lane = 5 
        speed_limit = 105 

        # super().__init__(num_env, num_auto_vehicle, num_idm_vehicle, num_lane, speed_limit, no_steering, device)

        self.pacecar_env = True

        # Get info from csv file
        self.csv_path = csv_path 
        self.process_csv()

    def process_csv(self):
        
        print("loading ngsim dataset from csv: %s" % (self.csv_path))

        df = pd.read_csv(self.csv_path, header=0, sep=',')
        df = df[df.v_Class != 1]
        df = df.drop(columns=["Global_Time", "v_Width", "Global_X", "Global_Y", "v_Class"])

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
        print(self.df.head(10))

    def reset(self):

        super().reset()

        # Make road network
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

        # allocate idm vehicles;
        num_idm_vehicle_per_lane = []
        for i in range(self.num_lane):
            num_idm_vehicle_per_lane.append(self.num_idm_vehicle // self.num_lane)

        num_remain_idm_vehicle = self.num_idm_vehicle - (self.num_idm_vehicle // self.num_lane) * self.num_lane
        for i in range(self.num_lane):
            if num_remain_idm_vehicle == 0:
                break
            num_idm_vehicle_per_lane[i] += 1
            num_remain_idm_vehicle -= 1

        # allocate auto vehicles;
        num_auto_vehicle_per_lane = []
        for i in range(self.num_lane):
            num_auto_vehicle_per_lane.append(self.num_auto_vehicle // self.num_lane)

        num_remain_auto_vehicle = self.num_auto_vehicle - (self.num_auto_vehicle // self.num_lane) * self.num_lane
        for i in range(self.num_lane):
            if num_remain_auto_vehicle == 0:
                break
            num_auto_vehicle_per_lane[i] += 1
            num_remain_auto_vehicle -= 1

        # make idm vehicles;
        idm_vehicle_id = 0
        lane_max_pos = []
        for i in range(self.num_lane):
            num_idm_vehicle_on_curr_lane = num_idm_vehicle_per_lane[i]

            curr_lane_max_pos = 0
            for j in range(num_idm_vehicle_on_curr_lane):
                nv = MicroVehicle.default_micro_vehicle(self.speed_limit)
                nv.position = j * 2.0 * nv.length
                
                vid = self.idm_vehicle_id(idm_vehicle_id)
                self.vehicle_position[:, vid] = self.tensorize_value(nv.position)
                self.vehicle_speed[:, vid] = self.tensorize_value(nv.speed)
                self.vehicle_accel_max[:, vid] = self.tensorize_value(nv.accel_max)
                self.vehicle_accel_pref[:, vid] = self.tensorize_value(nv.accel_pref)
                self.vehicle_target_speed[:, vid] = self.tensorize_value(nv.target_speed)
                self.vehicle_min_space[:, vid] = self.tensorize_value(nv.min_space)
                self.vehicle_time_pref[:, vid] = self.tensorize_value(nv.time_pref)
                self.vehicle_lane_id[:, vid] = self.tensorize_value(i, dtype=th.int32)
                self.vehicle_length[:, vid] = self.tensorize_value(nv.length)

                curr_lane_max_pos = nv.position

                idm_vehicle_id += 1

            lane_max_pos.append(curr_lane_max_pos)

        self.update_info()

    def reset_env(self, env_id: List[int]):
        super().reset_env(env_id)

        with th.no_grad():
            # allocate idm vehicles;
            num_idm_vehicle_per_lane = []
            for i in range(self.num_lane):
                num_idm_vehicle_per_lane.append(self.num_idm_vehicle // self.num_lane)

            num_remain_idm_vehicle = self.num_idm_vehicle - (self.num_idm_vehicle // self.num_lane) * self.num_lane
            for i in range(self.num_lane):
                if num_remain_idm_vehicle == 0:
                    break
                num_idm_vehicle_per_lane[i] += 1
                num_remain_idm_vehicle -= 1

            # allocate auto vehicles;
            num_auto_vehicle_per_lane = []
            for i in range(self.num_lane):
                num_auto_vehicle_per_lane.append(self.num_auto_vehicle // self.num_lane)

            num_remain_auto_vehicle = self.num_auto_vehicle - (self.num_auto_vehicle // self.num_lane) * self.num_lane
            for i in range(self.num_lane):
                if num_remain_auto_vehicle == 0:
                    break
                num_auto_vehicle_per_lane[i] += 1
                num_remain_auto_vehicle -= 1

            # make idm vehicles;
            idm_vehicle_id = 0
            lane_max_pos = []
            for i in range(self.num_lane):
                num_idm_vehicle_on_curr_lane = num_idm_vehicle_per_lane[i]

                curr_lane_max_pos = 0
                for j in range(num_idm_vehicle_on_curr_lane):
                    nv = MicroVehicle.default_micro_vehicle(self.speed_limit)
                    nv.position = j * 2.0 * nv.length
                    
                    vid = self.idm_vehicle_id(idm_vehicle_id)
                    self.vehicle_position[env_id, vid] = self.tensorize_value(nv.position)
                    self.vehicle_speed[env_id, vid] = self.tensorize_value(nv.speed)
                    self.vehicle_accel_max[env_id, vid] = self.tensorize_value(nv.accel_max)
                    self.vehicle_accel_pref[env_id, vid] = self.tensorize_value(nv.accel_pref)
                    self.vehicle_target_speed[env_id, vid] = self.tensorize_value(nv.target_speed)
                    self.vehicle_min_space[env_id, vid] = self.tensorize_value(nv.min_space)
                    self.vehicle_time_pref[env_id, vid] = self.tensorize_value(nv.time_pref)
                    self.vehicle_lane_id[env_id, vid] = self.tensorize_value(i, dtype=th.int32)
                    self.vehicle_length[env_id, vid] = self.tensorize_value(nv.length)

                    curr_lane_max_pos = nv.position

                    idm_vehicle_id += 1

                lane_max_pos.append(curr_lane_max_pos)

            # make auto vehicles;
            auto_vehicle_id = 0
            for i in range(self.num_lane):
                e_lane = self.hw_lane[i]
                curr_lane_max_pos = lane_max_pos[i]
                num_auto_vehicle_on_curr_lane = num_auto_vehicle_per_lane[i]

                for j in range(num_auto_vehicle_on_curr_lane):
                    longitudinal = curr_lane_max_pos + 50.0 + 10.0 * j
                    pos = e_lane.position(longitudinal, 0.0)
                    heading = e_lane.heading_at(longitudinal)
                    nv = Vehicle(self.hw_road, pos, heading, 0)                

                    vid = self.auto_vehicle_id(auto_vehicle_id)
                    self.vehicle_world_position[env_id, vid] = self.tensorize_value(pos)
                    self.vehicle_world_heading[env_id, vid] = self.tensorize_value(heading)
                    self.vehicle_world_velocity[env_id, vid] = self.tensorize_value(nv.velocity)
                    self.vehicle_speed[env_id, vid] = 0.
                    self.vehicle_length[env_id, vid] = self.tensorize_value(nv.LENGTH)
                    
                    auto_vehicle_id += 1

            self.update_info()