import os, sys
from typing import List

sys.path.append("../")

from envs.traffic._simulation import ParallelTrafficSim
from envs.traffic._idm import IDMLayer
from highway_env.road.lane import AbstractLane, LineType
from highway_env.vehicle.kinematics import Vehicle
from externals.traffic.road.vehicle.micro_vehicle import MicroVehicle
import numpy as np
import torch as th

class WorldSim(ParallelTrafficSim):

    def __init__(self, num_env: int, num_idm_vehicle: int, no_steering: bool=True, device="cpu"):

        super().__init__(num_env, 0, num_idm_vehicle, 5, 29.0576, no_steering, device)
        self.pacecar_env = True
        # self.init_position, self.init_speed = init_state # [num_env, 2] and [num_env, 1]
        self.initial_state = None 
        self.delta_time = 0.1
        

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

        self.init_position = init_position 
        self.init_speed = init_speed 
        self.lane_ids = lane_ids

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

        self.random_initial_state()

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
            min_size = min(acc.shape[-1], noise.shape[-1])
            acc = acc.clone()[:min_size] + noise[:min_size]

        # 3. update pos and vel of vehicles;
        self.vehicle_position[self.active_envs, self.num_auto_vehicle:] = (self.vehicle_position.clone() + self.vehicle_speed.clone() * self.delta_time)[:, self.num_auto_vehicle:]
        self.vehicle_speed[self.active_envs, self.num_auto_vehicle:] = (self.vehicle_speed[:, self.num_auto_vehicle:].clone() + acc * self.delta_time)

        # 6. move idm vehicles from lane to lane;
        self.update_idm_vehicle_lane_membership()

        self.update_info()

    def reset_env(self, env_id: List[int]):
        super().reset_env(env_id)
        self.active_envs = self.active_envs[self.active_envs!=self.active_envs[env_id]]