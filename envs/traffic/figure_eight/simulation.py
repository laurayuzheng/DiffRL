from typing import List
from envs.traffic._simulation import ParallelTrafficSim
from highway_env.road.lane import AbstractLane, LineType
from highway_env.vehicle.kinematics import Vehicle
from externals.traffic.road.vehicle.micro_vehicle import MicroVehicle
from envs.traffic.diff_highway_env.lane import *
import numpy as np
import torch as th

class FigureEightSim(ParallelTrafficSim):

    def __init__(self, num_env: int, num_auto_vehicle: int, num_idm_vehicle: int, num_lane: int, speed_limit: float, no_steering: bool, device):

        self.ring_lanes = 6 

        super().__init__(num_env, num_auto_vehicle, num_idm_vehicle, self.ring_lanes, speed_limit, no_steering, device)

    def reset(self):

        super().reset()

        radius = 30. # radius of figure eight rings
        transform = np.array([-1, 1]) # this is to transform flow coordinates --> highway_env

        # add intersection lanes; 
        center = np.array([0, 0]) * transform
        right = np.array([radius, 0]) * transform
        top = np.array([0, radius]) * transform
        left = np.array([-radius, 0]) * transform
        bottom = np.array([0, -radius]) * transform
        
        line_type = [LineType.CONTINUOUS, LineType.CONTINUOUS]
        self.make_straight_lane(0, left, center, "left", "center", line_type)
        self.make_straight_lane(1, center, right, "center", "right", line_type)
        self.make_straight_lane(2, top, center, "top", "center", line_type)
        self.make_straight_lane(3, center, bottom, "center", "bottom", line_type)

        # add circular lanes;
        
        sp4, ep4 = 270, 0  # counter clockwise; decreasing
        sp5, ep5 = -180, 90  # clockwise; increasing

        sp4, ep4 = np.deg2rad(sp4), np.deg2rad(ep4)
        sp5, ep5 = np.deg2rad(sp5), np.deg2rad(ep5)

        self.make_circular_lane(4, 
                                center + np.array([radius, radius]) * transform, 
                                radius, 
                                sp4, 
                                ep4, 
                                False, 
                                "right", 
                                "top",
                                line_type)
        self.make_circular_lane(5, 
                                center - np.array([radius, radius]) * transform, 
                                radius, 
                                sp5, 
                                ep5, 
                                True, 
                                "bottom", 
                                "left", 
                                line_type)


        self.next_lane[0] = [1]
        self.next_lane[1] = [4]
        self.next_lane[4] = [2]
        self.next_lane[2] = [3]
        self.next_lane[3] = [5]
        self.next_lane[5] = [0]

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
                auto_starting_lanes = [th.randint(0, self.ring_lanes-1, [1]).cpu().item()]

                # idm vehicles only start from border straight lanes;
                idm_starting_lanes = list(range(self.ring_lanes))
                idm_starting_lanes.remove(auto_starting_lanes[0])
                num_idm_starting_lanes = len(idm_starting_lanes)

                num_idm_vehicle_per_lane = []
                for i in range(num_idm_starting_lanes):
                    num_idm_vehicle_per_lane.append(self.num_idm_vehicle // num_idm_starting_lanes)

                num_remain_idm_vehicle = self.num_idm_vehicle - (self.num_idm_vehicle // num_idm_starting_lanes) * num_idm_starting_lanes
                
                # @sanghyun: also select idm lanes randomly;
                for i in th.randperm(num_idm_starting_lanes).cpu():
                    if num_remain_idm_vehicle == 0:
                        break
                    num_idm_vehicle_per_lane[i] += 1
                    num_remain_idm_vehicle -= 1

                # allocate auto vehicles;

                # auto vehicles only start from sine lanes;
                # auto_starting_lanes = [7]
                num_auto_starting_lanes = len(auto_starting_lanes)

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
                        
                        # assert nv.position < curr_lane_length, "Please reduce number of IDM vehicles, lane overflow!"
                        
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
