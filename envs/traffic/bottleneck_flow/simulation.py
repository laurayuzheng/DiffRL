from typing import List
from envs.traffic._simulation import ParallelTrafficSim
from highway_env.road.lane import AbstractLane, LineType
from highway_env.vehicle.kinematics import Vehicle
from externals.traffic.road.vehicle.micro_vehicle import MicroVehicle
from envs.traffic.diff_highway_env.lane import *
import numpy as np
import torch as th

class BottleneckSim(ParallelTrafficSim):

    def __init__(self, num_env: int, num_auto_vehicle: int, num_idm_vehicle: int, num_lane: int, speed_limit: float, no_steering: bool, device):

        num_lane = 8

        super().__init__(num_env, num_auto_vehicle, num_idm_vehicle, num_lane, speed_limit, no_steering, device)
        
    def reset(self):

        super().reset()
        
        curr_num_roads = 0
        num_parallel_lanes = 5
        num_bottleneck_lanes = 3

        # add highway lanes 
        ends = [150] # before, converging, merge, after
        self.auto_start_lane_id = len(ends) * 2 
        # merge_lane_id = 2

        lane_width = AbstractLane.DEFAULT_WIDTH

        # 5 lanes prior to bottleneck
        for i in range(num_parallel_lanes): # self.num_lanes 
            
            next_start = 0

            for j, length in enumerate(ends):
                
                start = np.array([next_start, -1 * float(i) * lane_width])
                end = np.array([next_start + length, -1 * float(i) * lane_width])
                next_start += length
                
                line_type = [LineType.STRIPED, LineType.STRIPED]
                if i == 0:
                    line_type[1] = LineType.CONTINUOUS
                        
                if i == num_parallel_lanes - 1:
                    line_type[0] = LineType.CONTINUOUS

                self.make_straight_lane(curr_num_roads, start, end, "straight_{}".format(curr_num_roads), "straight_{}".format((curr_num_roads + 1) % len(ends) + i*len(ends)), line_type)

                # self.add_next_lane(curr_num_roads, (curr_num_roads + 1) % len(ends) + i*len(ends))

                curr_num_roads += 1

        ends = [150]

        for i in range(num_bottleneck_lanes): # self.num_lanes 
            
            next_start = 150

            for j, length in enumerate(ends):
                
                start = np.array([next_start, -1 * float(i+1) * lane_width])
                end = np.array([next_start + length, -1 * float(i+1) * lane_width])
                next_start += length
                
                line_type = [LineType.STRIPED, LineType.STRIPED]
                if i == 0:
                    line_type[1] = LineType.CONTINUOUS
                        
                if i == num_bottleneck_lanes - 1: # self.num_lanes - 1
                    line_type[0] = LineType.CONTINUOUS

                self.make_straight_lane(curr_num_roads, start, end, "straight_{}".format(curr_num_roads), "straight_{}".format((curr_num_roads + 1) % len(ends) + i*len(ends)), line_type)
                
                # self.add_next_lane(curr_num_roads, (curr_num_roads + 1) % len(ends) + i*len(ends))

                curr_num_roads += 1
    
        self.add_next_lane(0, 5)
        self.add_next_lane(1, 5)
        self.add_next_lane(2, 6)
        self.add_next_lane(3, 7)
        self.add_next_lane(4, 7)

        self.add_next_lane(5, 0)
        self.add_next_lane(5, 1)
        self.add_next_lane(6, 2)
        self.add_next_lane(7, 3)
        self.add_next_lane(7, 4)

        self.fill_next_lane_tensor()

        # create vehicles;
        env_id = list(range(self.num_env))

        self.reset_env(env_id)

    def reset_env(self, env_id: List[int]):
        super().reset_env(env_id)

        with th.no_grad():
            
            for eid in env_id:
                    
                # @sanghyun: select starting lane of auto vehicle randomly;
                # this is needed to accelerate (generalized) training;
                auto_starting_lanes = [th.randint(self.auto_start_lane_id, self.auto_start_lane_id+2, [1]).cpu().item()]

                # idm vehicles only start from border straight lanes;
                idm_starting_lanes = list(range(self.num_lane))

                try:
                    if self.num_auto_vehicle > 0:
                        idm_starting_lanes.remove(auto_starting_lanes[0])
                except: 
                    pass 

                num_idm_starting_lanes = len(idm_starting_lanes)

                num_idm_vehicle_per_lane = []

                network_sum = 0 
                for i in range(num_idm_starting_lanes):
                    length = self.lane_length[idm_starting_lanes[i]]
                    network_sum += length
                
                vps = 0 
                for i in range(num_idm_starting_lanes):
                    lane_length = self.lane_length[idm_starting_lanes[i]]
                    p = float(lane_length / float(network_sum))
                    vp = int(p * self.num_idm_vehicle)
                    num_idm_vehicle_per_lane.append(vp)
                    vps += vp
                    # num_idm_vehicle_per_lane.append(self.num_idm_vehicle // num_idm_starting_lanes)

                num_remain_idm_vehicle = self.num_idm_vehicle - vps

                # @sanghyun: also select idm lanes randomly;
                while num_remain_idm_vehicle > 0: 
                    i = th.randint(0, num_idm_starting_lanes, [1]).cpu().item()
                    num_idm_vehicle_per_lane[i] += 1
                    num_remain_idm_vehicle -= 1 
                    vps += 1
                
                # Sanity check
                assert vps == self.num_idm_vehicle, "vps != num idm, vps = {}, num idm = {}".format(vps, self.num_idm_vehicle)

                # allocate auto vehicles;

                # auto vehicles only start from sine lanes;
                # auto_starting_lanes = [7]
                num_auto_starting_lanes = len(auto_starting_lanes)

                assert num_auto_starting_lanes == 1, "number of auto starting lanes is not 1"
                # assert self.num_auto_vehicle == 1, "should be at least one auto vehicle"

                num_auto_vehicle_per_lane = []
                for i in range(num_auto_starting_lanes):
                    num_auto_vehicle_per_lane.append(self.num_auto_vehicle // num_auto_starting_lanes)

                num_remain_auto_vehicle = self.num_auto_vehicle - (self.num_auto_vehicle // num_auto_starting_lanes) * num_auto_starting_lanes
                for i in range(num_auto_starting_lanes):
                    if num_remain_auto_vehicle == 0:
                        break
                    num_auto_vehicle_per_lane[i] += 1
                    num_remain_auto_vehicle -= 1

                # make idm vehicles;
                idm_vehicle_id = 0
                for i in range(num_idm_starting_lanes):
                    curr_lane_id = idm_starting_lanes[i]
                    curr_lane_length = self.lane_length[curr_lane_id].cpu().item()
                    num_idm_vehicle_on_curr_lane = num_idm_vehicle_per_lane[i]

                    for j in range(0, num_idm_vehicle_on_curr_lane):
                        nv = MicroVehicle.default_micro_vehicle(self.speed_limit)
                        nv.position = j * 2.0 * nv.length
                        
                        assert nv.position < curr_lane_length, "Please reduce number of IDM vehicles, lane overflow!"
                        
                        vid = self.idm_vehicle_id(idm_vehicle_id)
                        self.vehicle_position[eid, vid] = self.tensorize_value(nv.position)
                        self.vehicle_speed[eid, vid] = self.tensorize_value(nv.speed)
                        self.vehicle_accel_max[eid, vid] = self.tensorize_value(nv.accel_max)
                        self.vehicle_accel_pref[eid, vid] = self.tensorize_value(nv.accel_pref)
                        self.vehicle_target_speed[eid, vid] = self.tensorize_value(nv.target_speed)
                        self.vehicle_min_space[eid, vid] = self.tensorize_value(nv.min_space)
                        self.vehicle_time_pref[eid, vid] = self.tensorize_value(nv.time_pref)
                        self.vehicle_lane_id[eid, vid] = self.tensorize_value(curr_lane_id, dtype=th.int32)
                        self.vehicle_length[eid, vid] = self.tensorize_value(nv.length)

                        idm_vehicle_id += 1

                # make auto vehicles;
                auto_vehicle_id = 0
                for i in range(num_auto_starting_lanes):
                    curr_lane_id = auto_starting_lanes[i]
                    curr_lane_length = self.lane_length[curr_lane_id].cpu().item()
                    num_auto_vehicle_on_curr_lane = num_auto_vehicle_per_lane[i]

                    for j in range(num_auto_vehicle_on_curr_lane):
                        longitudinal = 10.0 * j

                        vid = self.auto_vehicle_id(auto_vehicle_id)
                        self.vehicle_position[eid, vid] = self.tensorize_value(longitudinal)
                        self.vehicle_speed[eid, vid] = self.tensorize_value(0.)
                        self.vehicle_lane_id[eid, vid] = self.tensorize_value(curr_lane_id, dtype=th.int32)
                        self.vehicle_length[eid, vid] = self.tensorize_value(Vehicle.LENGTH)
                        
                        auto_vehicle_id += 1

            if not self.no_steering:
                self.update_auto_world_info(env_id)

            self.update_info()
