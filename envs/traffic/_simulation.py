from time import time
from typing import Dict, List
from highway_env.road.road import RoadNetwork as hwRoadNetwork
from highway_env.road.regulation import RegulatedRoad as hwRoad
from highway_env.road.lane import LineType
from highway_env.road.lane import StraightLane, AbstractLane
import numpy as np
from envs.traffic._idm import IDMLayer
from envs.traffic.diff_highway_env.lane import *
from envs.traffic.diff_highway_env.kinematics import *

import torch as th

class ParallelTrafficSim:

    def __init__(self, num_env: int, num_auto_vehicle: int, num_idm_vehicle: int, num_lane: int, speed_limit: float, device):

        self.num_env = num_env
        self.num_vehicle = num_auto_vehicle + num_idm_vehicle
        self.num_auto_vehicle = num_auto_vehicle
        self.num_idm_vehicle = num_idm_vehicle
        self.num_lane = num_lane
        self.speed_limit = speed_limit
        self.device = device

        self.reset()

    def reset(self):

        self.hw_network = hwRoadNetwork()
        self.hw_road = hwRoad(self.hw_network, None)
        self.hw_lane: Dict[int, AbstractLane] = {}

        self.next_lane: Dict[int, List[int]] = {}
        self.next_lane_tensor = th.zeros((self.num_lane, self.num_lane), dtype=th.int32, device=self.device)
        
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
        
        # straight lane info;
        self.straight_lane_start = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)
        self.straight_lane_end = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)
        self.straight_lane_heading = th.zeros((self.num_lane,), dtype=th.float32, device=self.device)
        self.straight_lane_direction = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)
        self.straight_lane_direction_lateral = th.zeros((self.num_lane, 2), dtype=th.float32, device=self.device)

    def reset_env(self, env_id):
        with th.no_grad():
            self.vehicle_position[env_id, :] = 0.
            self.vehicle_speed[env_id, :] = 0.
            self.vehicle_accel_max[env_id, :] = 0.
            self.vehicle_accel_pref[env_id, :] = 0.
            self.vehicle_lane_id[env_id, :] = 0
            self.vehicle_length[env_id, :] = 0.
            self.vehicle_min_space[env_id, :] = 0.
            self.vehicle_target_speed[env_id, :] = 0.
            self.vehicle_time_pref[env_id, :] = 0.

            self.vehicle_pos_delta[env_id, :] = 0.
            self.vehicle_speed_delta[env_id, :] = 0.

            self.vehicle_world_heading[env_id, :] = 0.
            self.vehicle_world_position[env_id] = 0.
            self.vehicle_world_velocity[env_id] = 0. 

    def fill_next_lane_tensor(self):

        for i in range(self.num_lane):
            next_lanes = self.next_lane[i]
            j = 0
            while j < self.num_lane:
                self.next_lane_tensor[i, j] = next_lanes[j % len(next_lanes)]
                j += 1

    def auto_vehicle_id(self, id: int):

        return id

    def idm_vehicle_id(self, id: int):

        return self.num_auto_vehicle + id

    def update_info(self):

        self.update_idm_world_info()
        self.update_auto_local_info()

    def update_idm_world_info(self):

        '''
        Update world position, velocity, heading of idm vehicles.
        '''
        
        idm_idx = self.idm_vehicle_id(0)

        idm_longitudinal = self.vehicle_position[:, idm_idx:].clone()
        idm_lateral = th.zeros_like(idm_longitudinal)
        idm_speed = self.vehicle_speed[:, idm_idx:].clone()

        idm_longitudinal = idm_longitudinal.reshape((-1,))
        idm_lateral = idm_lateral.reshape((-1,))
        idm_speed = idm_speed.reshape((-1,))

        idm_world_position, idm_world_velocity, idm_world_heading = \
            straight_lane_position_velocity(idm_longitudinal, 
                                                idm_lateral, 
                                                idm_speed, 
                                                self.straight_lane_start,
                                                self.straight_lane_end,
                                                self.straight_lane_heading,
                                                self.straight_lane_direction,
                                                self.straight_lane_direction_lateral)

        idm_world_position = idm_world_position.view(self.num_env, self.num_idm_vehicle, self.num_lane, 2)
        idm_world_velocity = idm_world_velocity.view(self.num_env, self.num_idm_vehicle, self.num_lane, 2)
        idm_world_heading = idm_world_heading.view(self.num_env, self.num_idm_vehicle, self.num_lane)

        idm_lane_id = self.vehicle_lane_id[:, idm_idx:].clone()
        idm_lane_id_A = idm_lane_id.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, -1, 2)).to(dtype=th.int64)
        idm_lane_id_B = idm_lane_id.unsqueeze(-1).to(dtype=th.int64)
        
        sel_idm_world_position = th.gather(idm_world_position, 2, idm_lane_id_A).squeeze(2)
        sel_idm_world_velocity = th.gather(idm_world_velocity, 2, idm_lane_id_A).squeeze(2)
        sel_idm_world_heading = th.gather(idm_world_heading, 2, idm_lane_id_B).squeeze(2)

        self.vehicle_world_position[:, idm_idx:] = sel_idm_world_position
        self.vehicle_world_velocity[:, idm_idx:] = sel_idm_world_velocity
        self.vehicle_world_heading[:, idm_idx:] = sel_idm_world_heading

        return

    def update_auto_local_info(self):

        '''
        Update local position, lane id of auto vehicles.
        '''

        idm_idx = self.idm_vehicle_id(0)

        auto_world_position = self.vehicle_world_position[:, :idm_idx, :].clone()
        auto_world_position = auto_world_position.reshape((-1, 2))

        auto_distance, auto_longitudinal, _ = \
                                    straight_lane_distance(
                                        auto_world_position, 
                                        self.straight_lane_start,
                                        self.straight_lane_end, 
                                        self.straight_lane_heading,
                                        self.straight_lane_direction, 
                                        self.straight_lane_direction_lateral)

        auto_distance = auto_distance.view(self.num_env, self.num_auto_vehicle, self.num_lane)
        auto_longitudinal = auto_longitudinal.view(self.num_env, self.num_auto_vehicle, self.num_lane)
        
        min_auto_distance, min_auto_distance_idx = auto_distance.min(dim=2)
        
        min_auto_distance_idx_A = min_auto_distance_idx.unsqueeze(-1).to(th.int64)
        sel_auto_longitudinal = th.gather(auto_longitudinal, dim=2, index=min_auto_distance_idx_A).squeeze(-1)
        
        min_auto_distance_idx = th.where(min_auto_distance < AbstractLane.DEFAULT_WIDTH, 
                                            min_auto_distance_idx,
                                            -1)

        tmp_idx = th.where(min_auto_distance > AbstractLane.DEFAULT_WIDTH)
        sel_auto_longitudinal[tmp_idx] = 1000.0
        # sel_auto_longitudinal = th.where(min_auto_distance < AbstractLane.DEFAULT_WIDTH,
        #                                     sel_auto_longitudinal,
        #                                     1000.0) # some very large value so that it affects nothing;
        
        self.vehicle_lane_id[:, :idm_idx] = min_auto_distance_idx
        self.vehicle_position[:, :idm_idx] = sel_auto_longitudinal
        
        return

    def update_delta(self):

        '''
        Update [pos_delta] and [vel_delta].
        '''

        INF = 1e7

        vehicle_position = self.vehicle_position.clone()
        vehicle_lane_id = self.vehicle_lane_id.clone()
        vehicle_speed = self.vehicle_speed.clone()
        vehicle_length = self.vehicle_length.clone()

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
        vehicle_pos_delta = th.clip(vehicle_pos_delta, min=1e-3)
        vehicle_vel_delta = vehicle_speed - leading_vehicle_speed

        expanded_lane_length = self.lane_length.unsqueeze(0).expand((self.num_env, -1))
        vehicle_lane_length = th.gather(expanded_lane_length, 1, vehicle_lane_id.to(dtype=th.int64))
        vehicle_max_pos_delta = vehicle_lane_length - vehicle_position - vehicle_length * 0.5 + 10.
        
        self.vehicle_pos_delta = vehicle_pos_delta
        self.vehicle_speed_delta = vehicle_vel_delta

        tmp_idx = th.where(vehicle_pos_delta > vehicle_max_pos_delta)
        self.vehicle_pos_delta[tmp_idx] = vehicle_max_pos_delta[tmp_idx]
        self.vehicle_speed_delta[tmp_idx] = 0.
        
    def update_idm_vehicle_lane_membership(self):

        '''
        Update [position] and [lane_id].
        '''
        vehicle_lane_id = self.vehicle_lane_id.clone()
        vehicle_position = self.vehicle_position.clone()

        expanded_lane_length = self.lane_length.unsqueeze(0).expand((self.num_env, -1))
        vehicle_lane_length = th.gather(expanded_lane_length, 1, vehicle_lane_id.to(dtype=th.int64))
        
        tmp_idx = th.where(vehicle_position > vehicle_lane_length)
        self.vehicle_position[tmp_idx] = (vehicle_position[tmp_idx] - vehicle_lane_length[tmp_idx])

        curr_next_lane = self.next_lane_tensor[:, th.randint(self.num_lane, (1,)).cpu().item()]
        curr_next_lane = curr_next_lane.unsqueeze(0).expand((self.num_env, -1))
        curr_next_vehicle_lane_id = th.gather(curr_next_lane, 1, vehicle_lane_id.to(dtype=th.int64))

        self.vehicle_lane_id[tmp_idx] = curr_next_vehicle_lane_id[tmp_idx]

    def update_auto_vehicle(self, actions: th.Tensor, dt: float):

        steering = actions[:, 0::2]
        acceleration = actions[:, 1::2]

        steering = steering.reshape((-1,))
        acceleration = acceleration.reshape((-1,))

        position = self.vehicle_world_position[:, :self.num_auto_vehicle, :].reshape((self.num_env * self.num_auto_vehicle, -1)).clone()
        speed = self.vehicle_speed[:, :self.num_auto_vehicle].reshape((-1,)).clone()
        heading = self.vehicle_world_heading[:, :self.num_auto_vehicle].reshape((-1,)).clone()
        length = self.vehicle_length[:, :self.num_auto_vehicle].reshape((-1,)).clone()

        np, nv, nh, ns = auto_vehicle_apply_action(position, speed, heading, length, steering, acceleration, dt)
        
        self.vehicle_world_position[:, :self.num_auto_vehicle, :] = np.reshape((self.num_env, self.num_auto_vehicle, -1))
        self.vehicle_world_velocity[:, :self.num_auto_vehicle, :] = nv.reshape((self.num_env, self.num_auto_vehicle, -1))
        self.vehicle_world_heading[:, :self.num_auto_vehicle] = nh.reshape((self.num_env, self.num_auto_vehicle))
        self.vehicle_speed[:, :self.num_auto_vehicle] = ns.reshape((self.num_env, self.num_auto_vehicle))

        return
    
    def forward(self, actions: th.Tensor, delta_time: float):

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

        acc = IDMLayer.apply(accel_max, accel_pref, speed, target_speed, pos_delta, speed_delta, min_space, time_pref, delta_time)

        # 3. update pos and vel of vehicles;
        self.vehicle_position[:, self.num_auto_vehicle:] = (self.vehicle_position.clone() + self.vehicle_speed.clone() * delta_time)[:, self.num_auto_vehicle:]
        self.vehicle_speed[:, self.num_auto_vehicle:] = (self.vehicle_speed[:, self.num_auto_vehicle:].clone() + acc * delta_time)

        # 6. move idm vehicles from lane to lane;
        self.update_idm_vehicle_lane_membership()
        
        # 7. apply action and update state of auto vehicles;
        self.update_auto_vehicle(actions, delta_time)    

        self.update_info()

    def check_auto_collision(self):

        '''
        Return True if there is any auto vehicle that collides with IDM vehicle.
        '''

        with th.no_grad():

            vehicle_pos_a = self.vehicle_world_position.unsqueeze(0).unsqueeze(0).expand((self.num_env, self.num_vehicle, -1, -1, -1))
            vehicle_pos_b = self.vehicle_world_position.unsqueeze(2).unsqueeze(2).expand((-1, -1, self.num_env, self.num_vehicle, -1))

            pairwise_dist = vehicle_pos_a - vehicle_pos_b
            pairwise_dist = th.diagonal(pairwise_dist, dim1=0, dim2=2)
            pairwise_dist = th.transpose(pairwise_dist, 0, 3)
            pairwise_dist = th.transpose(pairwise_dist, 2, 3)
            pairwise_dist = th.norm(pairwise_dist, p=2, dim=-1)

            tmp_add = th.eye(self.num_vehicle, dtype=th.float32, device=self.device).unsqueeze(0) * 100.
            pairwise_dist = pairwise_dist + tmp_add - 2.5
            auto_dist = pairwise_dist[:, :self.num_auto_vehicle, :]
            auto_dist = auto_dist.reshape((self.num_env, -1))

            collided = th.any(auto_dist < 0., dim=1)
            
        return collided    

    
    def check_auto_outoflane(self):

        '''
        Return True if there is any auto vehicle that does not belong to any lane.
        '''

        with th.no_grad():

            auto_lane_id = self.vehicle_lane_id[:, :self.num_auto_vehicle]
            outoflane = th.any(auto_lane_id == -1, dim=1)
            
        return outoflane
    
    def clear_grad(self):
        with th.no_grad():
            for k, v in self.__dict__.items():
                if isinstance(v, th.Tensor):
                    self.__dict__[k] = v.detach()

    '''
    Lane functions.
    '''
    def make_straight_lane(self, 
                            lane_id: int, 
                            start_pos: np.ndarray, 
                            end_pos: np.ndarray, 
                            start_node: str, 
                            end_node: str,
                            line_types = None):

        if line_types == None:
            line_types = [LineType.CONTINUOUS, LineType.CONTINUOUS]
        
        # highway env lane;
        env_lane = StraightLane(start_pos, end_pos, line_types=line_types)
        self.hw_road.network.add_lane(start_node, end_node, env_lane)
        self.hw_lane[lane_id] = env_lane
        
        # lane info;
        self.straight_lane_start[lane_id] = self.tensorize_value(env_lane.start)
        self.straight_lane_end[lane_id] = self.tensorize_value(env_lane.end)
        self.straight_lane_direction[lane_id] = self.tensorize_value(env_lane.direction)
        self.straight_lane_direction_lateral[lane_id] = self.tensorize_value(env_lane.direction_lateral)
        self.straight_lane_heading[lane_id] = self.tensorize_value(env_lane.heading)

        return lane_id

    def tensorize_value(self, value: np.ndarray, dtype=th.float32):
        return th.tensor(value, dtype=dtype, device=self.device)

    def add_vehicles_to_road_for_view(self):

        with th.no_grad():
            env_id = 0

            self.hw_road.vehicles.clear()

            observer_vehicle = None
            for i in range(self.num_vehicle):
                pos = self.vehicle_world_position[env_id, i].cpu().numpy()
                heading = self.vehicle_world_heading[env_id, i].cpu().numpy()
                rv = Vehicle(self.hw_road, pos, heading)

                self.hw_road.vehicles.append(rv)

                if observer_vehicle == None:
                    observer_vehicle = rv
            
            return observer_vehicle

