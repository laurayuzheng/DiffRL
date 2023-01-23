from typing import List
from envs.traffic._simulation import ParallelTrafficSim
from highway_env.road.lane import AbstractLane, LineType
from highway_env.vehicle.kinematics import Vehicle
from externals.traffic.road.vehicle.micro_vehicle import MicroVehicle
import numpy as np
import torch as th

class RoundaboutSim(ParallelTrafficSim):

    def __init__(self, num_env: int, num_auto_vehicle: int, num_idm_vehicle: int, num_lane: int, speed_limit: float, no_steering: bool, device):

        num_lane = 8 + 4 * 4

        super().__init__(num_env, num_auto_vehicle, num_idm_vehicle, num_lane, speed_limit, no_steering, device)

    def reset(self):

        super().reset()

        curr_num_lane = 0

        # add circular lanes;

        # only add outer circular lane for now;

        for lane_id in range(8):

            alpha = 24.
            radius = 24.

            if lane_id == 0:
                sp, ep = 90. - alpha, alpha
            elif lane_id == 1:
                sp, ep = alpha, -alpha
            elif lane_id == 2:
                sp, ep = -alpha, -90. + alpha
            elif lane_id == 3:
                sp, ep = -90. + alpha, -90. - alpha
            elif lane_id == 4:
                sp, ep = -90. - alpha, -180. + alpha
            elif lane_id == 5:
                sp, ep = -180. + alpha, -180. - alpha
            elif lane_id == 6:
                sp, ep = 180. - alpha, 90. + alpha
            else:
                sp, ep = 90. + alpha, 90. - alpha

            sp, ep = np.deg2rad(sp), np.deg2rad(ep)
            center = np.array([0., 0.])

            self.make_circular_lane(curr_num_lane, 
                                    center, 
                                    radius, 
                                    sp, 
                                    ep, 
                                    False, 
                                    "circular_{}".format(lane_id), 
                                    "circular_{}".format(lane_id + 1))

            self.add_next_lane(curr_num_lane, (curr_num_lane + 1) % 8)
            curr_num_lane += 1

        
        # add access lanes (straight, sine);

        for direction in ["south", "east", "north", "west"]:

            access = 170.
            dev = 85.
            a = 5.
            delta_st = 0.2 * dev

            delta_en = dev - delta_st
            w = 2. * np.pi / dev

            if direction == 'south':
                st_s0, st_e0, st_lt0 = [2., dev], [2., dev / 2], [LineType.STRIPED, LineType.CONTINUOUS]
                st_s1, st_e1, st_lt1 = [-2., dev / 2], [-2., dev], [LineType.NONE, LineType.CONTINUOUS]
                
                si_s0, si_e0, si_lt0 = [2. + a, dev / 2], [2. + a, dev / 2 - delta_st], [LineType.CONTINUOUS, LineType.CONTINUOUS]
                si_am0, si_pu0, si_ph0 = a, w, -np.pi / 2

                si_s1, si_e1, si_lt1 = [-2. - a, -dev / 2 + delta_en], [-2. - a, dev / 2], [LineType.CONTINUOUS, LineType.CONTINUOUS]
                si_am1, si_pu1, si_ph1 = a, w, -np.pi / 2 + w * delta_en
            elif direction == 'east':
                st_s0, st_e0, st_lt0 = [dev, -2.], [dev / 2, -2.], [LineType.STRIPED, LineType.CONTINUOUS]
                st_s1, st_e1, st_lt1 = [dev / 2., 2.], [dev, 2.], [LineType.NONE, LineType.CONTINUOUS]
                
                si_s0, si_e0, si_lt0 = [dev / 2, -2. - a], [dev / 2 - delta_st, -2. - a], [LineType.CONTINUOUS, LineType.CONTINUOUS]
                si_am0, si_pu0, si_ph0 = a, w, -np.pi / 2

                si_s1, si_e1, si_lt1 = [-dev / 2. + delta_en, 2. + a], [dev / 2, 2. + a], [LineType.CONTINUOUS, LineType.CONTINUOUS]
                si_am1, si_pu1, si_ph1 = a, w, -np.pi / 2 + w * delta_en

            elif direction == "north":
                st_s0, st_e0, st_lt0 = [-2., -dev], [-2., -dev / 2], [LineType.STRIPED, LineType.CONTINUOUS]
                st_s1, st_e1, st_lt1 = [2., -dev / 2.], [2., -dev], [LineType.NONE, LineType.CONTINUOUS]
                
                si_s0, si_e0, si_lt0 = [-2. - a, -dev / 2], [-2. - a, -dev / 2 + delta_st], [LineType.CONTINUOUS, LineType.CONTINUOUS]
                si_am0, si_pu0, si_ph0 = a, w, -np.pi / 2

                si_s1, si_e1, si_lt1 = [2. + a, dev / 2 - delta_en], [2. + a, -dev / 2], [LineType.CONTINUOUS, LineType.CONTINUOUS]
                si_am1, si_pu1, si_ph1 = a, w, -np.pi / 2 + w * delta_en

            else:
                st_s0, st_e0, st_lt0 = [-dev, 2.], [-dev / 2, 2.], [LineType.STRIPED, LineType.CONTINUOUS]
                st_s1, st_e1, st_lt1 = [-dev / 2, -2.], [-dev, -2.], [LineType.NONE, LineType.CONTINUOUS]
                
                si_s0, si_e0, si_lt0 = [-dev / 2, 2. + a], [-dev / 2 + delta_st, 2 + a], [LineType.CONTINUOUS, LineType.CONTINUOUS]
                si_am0, si_pu0, si_ph0 = a, w, -np.pi / 2

                si_s1, si_e1, si_lt1 = [dev / 2 - delta_en, -2 - a], [-dev / 2, -2 - a], [LineType.CONTINUOUS, LineType.CONTINUOUS]
                si_am1, si_pu1, si_ph1 = a, w, -np.pi / 2 + w * delta_en

            st0_id = self.make_straight_lane(curr_num_lane, st_s0, st_e0, "s00", "s01", st_lt0)
            curr_num_lane += 1
            st1_id = self.make_straight_lane(curr_num_lane, st_s1, st_e1, "s00", "s01", st_lt1)
            curr_num_lane += 1
            si0_id = self.make_sine_lane(curr_num_lane, si_s0, si_e0, si_am0, si_pu0, si_ph0, "s00", "s01", si_lt0)
            curr_num_lane += 1
            si1_id = self.make_sine_lane(curr_num_lane, si_s1, si_e1, si_am1, si_pu1, si_ph1, "s00", "s01", si_lt1)
            curr_num_lane += 1

            self.add_next_lane(st0_id, si0_id)
            self.add_next_lane(si1_id, st1_id)
            self.add_next_lane(st1_id, st0_id)
            if direction == 'south':
                self.add_next_lane(si0_id, 0)
                self.add_next_lane(6, si1_id)
            elif direction == 'east':
                self.add_next_lane(si0_id, 2)
                self.add_next_lane(0, si1_id)
            elif direction == 'north':
                self.add_next_lane(si0_id, 4)
                self.add_next_lane(2, si1_id)
            else:
                self.add_next_lane(si0_id, 6)
                self.add_next_lane(4, si1_id)
                
        self.fill_next_lane_tensor()

        # create vehicles;
        env_id = list(range(self.num_env))
        self.reset_env(env_id)

    def reset_env(self, env_id: List[int]):
        super().reset_env(env_id)

        with th.no_grad():
            # allocate idm vehicles;

            # idm vehicles only start from border straight lanes;
            idm_starting_lanes = [8, 9, 12, 13, 16, 17, 20, 21]
            num_idm_starting_lanes = len(idm_starting_lanes)

            num_idm_vehicle_per_lane = []
            for i in range(num_idm_starting_lanes):
                num_idm_vehicle_per_lane.append(self.num_idm_vehicle // num_idm_starting_lanes)

            num_remain_idm_vehicle = self.num_idm_vehicle - (self.num_idm_vehicle // num_idm_starting_lanes) * num_idm_starting_lanes
            for i in range(num_idm_starting_lanes):
                if num_remain_idm_vehicle == 0:
                    break
                num_idm_vehicle_per_lane[i] += 1
                num_remain_idm_vehicle -= 1

            # allocate auto vehicles;

            # auto vehicles only start from sine lanes;
            auto_starting_lanes = [10, 14, 18, 22]
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

                for j in range(num_idm_vehicle_on_curr_lane):
                    nv = MicroVehicle.default_micro_vehicle(self.speed_limit)
                    nv.position = j * 2.0 * nv.length
                    
                    assert nv.position < curr_lane_length, "Please reduce number of IDM vehicles, lane overflow!"
                    
                    vid = self.idm_vehicle_id(idm_vehicle_id)
                    self.vehicle_position[env_id, vid] = self.tensorize_value(nv.position)
                    self.vehicle_speed[env_id, vid] = self.tensorize_value(nv.speed)
                    self.vehicle_accel_max[env_id, vid] = self.tensorize_value(nv.accel_max)
                    self.vehicle_accel_pref[env_id, vid] = self.tensorize_value(nv.accel_pref)
                    self.vehicle_target_speed[env_id, vid] = self.tensorize_value(nv.target_speed)
                    self.vehicle_min_space[env_id, vid] = self.tensorize_value(nv.min_space)
                    self.vehicle_time_pref[env_id, vid] = self.tensorize_value(nv.time_pref)
                    self.vehicle_lane_id[env_id, vid] = self.tensorize_value(curr_lane_id, dtype=th.int32)
                    self.vehicle_length[env_id, vid] = self.tensorize_value(nv.length)

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
                    self.vehicle_position[env_id, vid] = self.tensorize_value(longitudinal)
                    self.vehicle_speed[env_id, vid] = self.tensorize_value(0.)
                    self.vehicle_lane_id[env_id, vid] = self.tensorize_value(curr_lane_id, dtype=th.int32)
                    self.vehicle_length[env_id, vid] = self.tensorize_value(Vehicle.LENGTH)
                    
                    auto_vehicle_id += 1

            self.update_info()