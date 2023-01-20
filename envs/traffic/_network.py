from typing import Dict, List
from externals.traffic.example.common._comp_lane import CompLane
from externals.traffic.road.network.road_network import RoadNetwork, DEFAULT_HEAD_POSITION_DELTA, DEFAULT_HEAD_SPEED_DELTA

from envs.traffic._lane import FastMicroLane
from envs.traffic.diff_highway_env.lane import *
from envs.traffic.diff_highway_env.kinematics import *

from highway_env.road.lane import AbstractLane
from highway_env.road.road import RoadNetwork as hwRoadNetwork
from highway_env.road.graphics import LineType
from highway_env.road.regulation import RegulatedRoad as hwRegulatedRoad

import numpy as np
import torch as th

class ParallelRoadNetwork(RoadNetwork):

    '''
    This class implements parallel version of road network for micro traffic simulation.
    '''

    def __init__(self, speed_limit: float, num_auto_vehicle: int, num_idm_vehicle: int, num_all_lane: int, device):

        super().__init__(speed_limit)

        self.num_auto_vehicle = num_auto_vehicle
        self.num_idm_vehicle = num_idm_vehicle
        self.num_all_lane = num_all_lane
        
        self.device = device

    def reset(self):

        super().__init__(self.speed_limit)

        # highway env road network that defines topology of the roads;
        # it will be used for rendering / collision detections;
        self.hw_road_network = hwRoadNetwork()
        self.hw_road = hwRegulatedRoad(self.hw_road_network)

        # comp lane;
        self.comp_lane: Dict[int, CompLane] = {}

        # straight lane;
        self.straight_lane_start = th.zeros((self.num_all_lane, 2), dtype=th.float32, device=self.device)
        self.straight_lane_end = th.zeros((self.num_all_lane, 2), dtype=th.float32, device=self.device)
        self.straight_lane_heading = th.zeros((self.num_all_lane,), dtype=th.float32, device=self.device)
        self.straight_lane_direction = th.zeros((self.num_all_lane, 2), dtype=th.float32, device=self.device)
        self.straight_lane_direction_lateral = th.zeros((self.num_all_lane, 2), dtype=th.float32, device=self.device)

        # auto vehicles;
        self.auto_vehicle_position = th.zeros((self.num_auto_vehicle, 2), dtype=th.float32, device=self.device)
        self.auto_vehicle_velocity = th.zeros((self.num_auto_vehicle, 2), dtype=th.float32, device=self.device)
        self.auto_vehicle_heading = th.zeros((self.num_auto_vehicle,), dtype=th.float32, device=self.device)
        self.auto_vehicle_speed = th.zeros((self.num_auto_vehicle,), dtype=th.float32, device=self.device)
        self.auto_vehicle_length = th.zeros((self.num_auto_vehicle,), dtype=th.float32, device=self.device)
        self.auto_vehicle_diagonal = th.zeros((self.num_auto_vehicle,), dtype=th.float32, device=self.device)

        '''
        Tensor Info.

        Below data is updated at the end of every forward call,
        and it is represented in world coordinates system.
        '''
        # idm vehicles;
        self.idm_vehicle_position = th.zeros((self.num_idm_vehicle, 2), dtype=th.float32, device=self.device)
        self.idm_vehicle_velocity = th.zeros((self.num_idm_vehicle, 2), dtype=th.float32, device=self.device)
        self.idm_vehicle_heading = th.zeros((self.num_idm_vehicle,), dtype=th.float32, device=self.device)

        # auto vehicle's local coordinates for each lane: [vehicle_id, lane_id];
        self.auto_vehicle_lane_longitudinal = th.zeros((self.num_auto_vehicle, self.num_all_lane), dtype=th.float32, device=self.device)
        self.auto_vehicle_lane_lateral = th.zeros((self.num_auto_vehicle, self.num_all_lane), dtype=th.float32, device=self.device)
        self.auto_vehicle_lane_distance = th.zeros((self.num_auto_vehicle, self.num_all_lane), dtype=th.float32, device=self.device)

    def forward_auto(self, steering: th.Tensor, acceleration: th.Tensor, dt: float):

        assert steering.ndim == 1 and acceleration.ndim == 1, ""
        assert len(steering) == self.num_auto_vehicle and len(acceleration) == self.num_auto_vehicle, ""

        np, nv, nh, ns = auto_vehicle_apply_action(self.auto_vehicle_position,
                                                    self.auto_vehicle_speed,
                                                    self.auto_vehicle_heading,
                                                    self.auto_vehicle_length,
                                                    steering,
                                                    acceleration,
                                                    dt)
        
        self.auto_vehicle_position = np
        self.auto_vehicle_velocity = nv
        self.auto_vehicle_heading = nh
        self.auto_vehicle_speed = ns

    def update_tensor_info(self):

        self.update_idm_tensor_info()

        self.update_auto_tensor_info()

    def update_idm_tensor_info(self):

        # collect longitudinal position and speed of idm vehicles;
        idm_curr_longitudinal = None
        idm_curr_speed = None
        idm_curr_vehicle_id = None
        
        for lane in self.lane.values():

            curr_longitudinal = lane.curr_pos
            curr_speed = lane.curr_vel
            curr_vehicle_id = lane.vehicle_id

            if idm_curr_longitudinal == None:
                idm_curr_longitudinal = curr_longitudinal
            else:
                idm_curr_longitudinal = th.cat([idm_curr_longitudinal, curr_longitudinal])

            if idm_curr_speed == None:
                idm_curr_speed = curr_speed
            else:
                idm_curr_speed = th.cat([idm_curr_speed, curr_speed])

            if idm_curr_vehicle_id == None:
                idm_curr_vehicle_id = curr_vehicle_id
            else:
                idm_curr_vehicle_id = th.cat([idm_curr_vehicle_id, curr_vehicle_id])

        # positions and velocities using straight lanes;
        idm_curr_lateral = th.zeros_like(idm_curr_longitudinal)
        idm_position, idm_velocity, idm_heading = straight_lane_position_velocity(idm_curr_longitudinal,
                                                                    idm_curr_lateral,
                                                                    idm_curr_speed,
                                                                    self.straight_lane_start,
                                                                    self.straight_lane_end,
                                                                    self.straight_lane_heading,
                                                                    self.straight_lane_direction,
                                                                    self.straight_lane_direction_lateral)

        # select corresponding lane info;
        for i in range(self.num_idm_vehicle):
            vid = idm_curr_vehicle_id[i]

            nr = self.micro_route[vid.cpu().item()]
            lane_id = nr.curr_lane_id()
            
            self.idm_vehicle_position[vid] = idm_position[i, lane_id]
            self.idm_vehicle_velocity[vid] = idm_velocity[i, lane_id]
            self.idm_vehicle_heading[vid] = idm_heading[i, lane_id]
        
        return

    def update_auto_tensor_info(self):

        '''
        Update following info about auto vehicles.

        * auto_vehicle_lane_longitudinal
        * auto_vehicle_lane_lateral
        * auto_vehicle_lane_distance
        '''

        longitudinal, lateral = straight_lane_local_coords(self.auto_vehicle_position,
                                                            self.straight_lane_start,
                                                            self.straight_lane_end,
                                                            self.straight_lane_heading,
                                                            self.straight_lane_direction,
                                                            self.straight_lane_direction_lateral)

        distance = straight_lane_distance(self.auto_vehicle_position,
                                            self.straight_lane_start,
                                            self.straight_lane_end,
                                            self.straight_lane_heading,
                                            self.straight_lane_direction,
                                            self.straight_lane_direction_lateral)

        self.auto_vehicle_lane_longitudinal = longitudinal
        self.auto_vehicle_lane_lateral = lateral
        self.auto_vehicle_lane_distance = distance - self.auto_vehicle_diagonal * 0.5

    def add_auto_vehicle(self, id: int, nv: Vehicle):

        self.auto_vehicle_position[id] = self.tensorize_value(nv.position)
        self.auto_vehicle_velocity[id] = self.tensorize_value(nv.velocity)
        self.auto_vehicle_speed[id] = self.tensorize_value(nv.speed)
        self.auto_vehicle_length[id] = self.tensorize_value(nv.LENGTH)
        self.auto_vehicle_heading[id] = self.tensorize_value(nv.heading)
        self.auto_vehicle_diagonal[id] = self.tensorize_value(nv.diagonal)
        
    def conversion(self, delta_time: float):

        for lane in self.lane.values():

            d_lane: FastMicroLane = lane

            if d_lane.num_vehicle():
                # check if head vehicle goes out of the lane;
                hv_curr_pos = d_lane.curr_pos[-1]                
                if hv_curr_pos >= d_lane.length:

                    vid = d_lane.get_head_vehicle_id()
                    hr = self.micro_route[vid]

                    next_lane_id = hr.next_lane_id()

                    if next_lane_id == -1:
                        # go back to first lane in the route;
                        hr.curr_idx = 0
                        next_lane_id = hr.curr_lane_id()
                    else:
                        hr.increment_curr_idx()

                    next_lane: FastMicroLane = self.lane[next_lane_id]

                    next_lane.add_tail_vehicle_tensor(d_lane.vehicle_id[-1],
                                                d_lane.curr_pos[-1] - d_lane.length,
                                                d_lane.curr_vel[-1],
                                                d_lane.accel_max[-1],
                                                d_lane.accel_pref[-1],
                                                d_lane.target_vel[-1],
                                                d_lane.min_space[-1],
                                                d_lane.time_pref[-1],
                                                d_lane.vehicle_length[-1])

                    d_lane.remove_head_vehicle()

    def resolve_next_state(self):

        pass

    def update_state(self):

        for lane in self.lane.values():

            lane.update_state()

    def update_next_state(self, acc: th.Tensor, delta_time: float):

        idx_start = 0
        
        for lane in self.lane.values():

            d_lane: FastMicroLane = lane
            idx_end = idx_start + d_lane.num_vehicle()

            curr_acc = acc[idx_start:idx_end]

            d_lane.next_pos = d_lane.curr_pos + d_lane.curr_vel * delta_time
            d_lane.next_vel = d_lane.curr_vel + curr_acc * delta_time

            idx_start = idx_end

    def tensorize(self, delta_time: float):

        a_max = []
        a_pref = []
        curr_vel = []
        target_vel = []
        pos_delta = []
        vel_delta = []
        min_space = []
        time_pref = []
        dt = []

        for lane in self.lane.values():

            d_lane: FastMicroLane = lane
            last_pos_delta, last_vel_delta = self.boundary(d_lane)
            curr_pos_delta, curr_vel_delta, curr_dt = d_lane.preprocess(last_pos_delta, last_vel_delta, delta_time)

            # check auto vehicles if they block the lane;
            for i in range(self.num_auto_vehicle):
                distance = self.auto_vehicle_lane_distance[i, lane.id]
                if distance < 0:
                    av_pos = self.auto_vehicle_lane_longitudinal[i, lane.id]
                    av_vel = self.auto_vehicle_speed[i]
                    av_len = self.auto_vehicle_length[i]

                    # if [av] is in front of the head vehicle, it would have been dealt with in [boundary];
                    for i in range(d_lane.num_vehicle() - 1):
                        curr_vpos_delta = th.clamp(av_pos - d_lane.curr_pos[i] - (d_lane.vehicle_length[i] + av_len) * 0.5, min=0.)
                        
                        if d_lane.curr_pos[i + 1] > av_pos and curr_vpos_delta < curr_pos_delta[i]:
                            curr_pos_delta[i] = curr_vpos_delta
                            curr_vel_delta[i] = d_lane.curr_vel[i] - av_vel
                        break
            
            a_max.append(d_lane.accel_max)
            a_pref.append(d_lane.accel_pref)
            curr_vel.append(d_lane.curr_vel)
            target_vel.append(d_lane.target_vel)
            pos_delta.append(curr_pos_delta)
            vel_delta.append(curr_vel_delta)
            min_space.append(d_lane.min_space)
            time_pref.append(d_lane.time_pref)
            dt.append(curr_dt)

        a_max = th.cat(a_max)
        a_pref = th.cat(a_pref)
        curr_vel = th.cat(curr_vel)
        target_vel = th.cat(target_vel)
        pos_delta = th.cat(pos_delta)
        vel_delta = th.cat(vel_delta)
        min_space = th.cat(min_space)
        time_pref = th.cat(time_pref)
        dt = th.cat(dt)

        return a_max, a_pref, curr_vel, target_vel, pos_delta, vel_delta, min_space, time_pref, dt

    def boundary(self, lane: FastMicroLane):

        '''
        Compute leading vehicle's pos delta and vel delta.
        '''

        position_delta = DEFAULT_HEAD_POSITION_DELTA
        speed_delta = DEFAULT_HEAD_SPEED_DELTA

        head_vehicle_id = lane.get_head_vehicle_id().cpu().item()

        if head_vehicle_id == -1:
            return position_delta, speed_delta
    
        v_pos = lane.curr_pos[-1]
        v_len = lane.vehicle_length[-1]
        v_speed = lane.curr_vel[-1]
        nr = self.micro_route[head_vehicle_id]

        # first find if there is auto vehicle on current lane that is ahead;
        for i in range(self.num_auto_vehicle):
            distance = self.auto_vehicle_lane_distance[i, nr.curr_lane_id()]
            if distance < 0:
                longitudinal = self.auto_vehicle_lane_longitudinal[i, nr.curr_lane_id()]
                if longitudinal > v_pos:
                    av_len = self.auto_vehicle_length[i]
                    av_speed = self.auto_vehicle_speed[i]

                    position_delta = longitudinal - v_pos - (v_len + av_len) * 0.5
                    position_delta = th.clamp(position_delta, min=0.)
                    speed_delta = v_speed - av_speed

                    return position_delta, speed_delta

        curr_head_position_delta = lane.length - v_pos - v_len * 0.5

        # iterate through future path and find leading vehicle;
        curr_lane_idx = nr.curr_idx + 1
        curr_lane_idx = curr_lane_idx % nr.route_length()

        while curr_lane_idx != nr.curr_idx:

            next_lane_id = nr.route[curr_lane_idx]
            next_lane: FastMicroLane = self.lane[next_lane_id]

            # find out the first av that is on the lane;
            overlap_av = None
            overlap_av_pos = 1e6
            for i in range(self.num_auto_vehicle):
                distance = self.auto_vehicle_lane_distance[i, next_lane_id]
                if distance < 0:
                    longitudinal = self.auto_vehicle_lane_longitudinal[i, next_lane_id]
                    if overlap_av is None or longitudinal < overlap_av_pos:
                        overlap_av = i
                        overlap_av_pos = longitudinal
                        
            if next_lane.num_vehicle() == 0:

                if overlap_av is None:
                
                    curr_lane_idx += 1
                    
                    # go back to first route;
                    if curr_lane_idx == nr.route_length():
                        curr_lane_idx = 0

                    continue

                else:

                    lv_pos = overlap_av_pos
                    lv_len = self.auto_vehicle_length[overlap_av]
                    lv_speed = self.auto_vehicle_speed[overlap_av]

            else:

                lv_pos = next_lane.curr_pos[0]
                lv_len = next_lane.vehicle_length[0]
                lv_speed = next_lane.curr_vel[0]

                if overlap_av_pos < lv_pos:

                    lv_pos = overlap_av_pos
                    lv_len = self.auto_vehicle_length[overlap_av]
                    lv_speed = self.auto_vehicle_speed[overlap_av]

            position_delta = curr_head_position_delta + (lv_pos - lv_len * 0.5)
            position_delta = th.clamp(position_delta, min=0.)
            speed_delta = v_speed - lv_speed

            break

        return position_delta, speed_delta

    def make_straight_lane(self, 
                            start_pos: np.ndarray, 
                            end_pos: np.ndarray, 
                            start_node: str, 
                            end_node: str,
                            line_types = None,
                            reverse = False):

        if line_types == None:
            line_types = [LineType.CONTINUOUS, LineType.CONTINUOUS]
        
        # highway env lane;
        env_lane = StraightLane(start_pos, end_pos, line_types=line_types, speed_limit=self.speed_limit)
        self.hw_road.network.add_lane(start_node, end_node, env_lane)

        # simulator lane;
        lane_length = env_lane.length
        sim_lane = FastMicroLane(lane_length, self.speed_limit, self.device)
        lane_id = self.add_lane(sim_lane)

        # comp lane;
        self.comp_lane[lane_id] = CompLane(self.hw_road, env_lane, sim_lane, reverse)

        # lane info;
        self.straight_lane_start[lane_id] = self.tensorize_value(env_lane.start)
        self.straight_lane_end[lane_id] = self.tensorize_value(env_lane.end)
        self.straight_lane_direction[lane_id] = self.tensorize_value(env_lane.direction)
        self.straight_lane_direction_lateral[lane_id] = self.tensorize_value(env_lane.direction_lateral)
        self.straight_lane_heading[lane_id] = self.tensorize_value(env_lane.heading)

        return lane_id

    def check_auto_collision(self):

        '''
        Return True if there is any auto vehicle that collides with IDM vehicle.
        '''

        with th.no_grad():

            entire_vehicle_pos = th.cat([self.auto_vehicle_position, self.idm_vehicle_position], dim=0)
            entire_vehicle_heading = th.cat([self.auto_vehicle_heading, self.idm_vehicle_heading], dim=0)
            num_entire_vehicle = len(entire_vehicle_pos)
            
            entire_vehicle_pos_a = entire_vehicle_pos.unsqueeze(1).expand([-1, num_entire_vehicle, -1])
            entire_vehicle_pos_b = entire_vehicle_pos.unsqueeze(0).expand([num_entire_vehicle, -1, -1])

            pairwise_dist = th.norm(entire_vehicle_pos_a - entire_vehicle_pos_b, dim=2)
            pairwise_dist = pairwise_dist - 15.

            for i in range(num_entire_vehicle):
                for j in range(i + 1, num_entire_vehicle, 1):
                    if pairwise_dist[i, j] < 0:
                        va = Vehicle(self.hw_road, entire_vehicle_pos[i].cpu().numpy(), entire_vehicle_heading[i].cpu().item())
                        vb = Vehicle(self.hw_road, entire_vehicle_pos[j].cpu().numpy(), entire_vehicle_heading[j].cpu().item())

                        collide, _, _ = va._is_colliding(vb, 0.)

                        if collide:
                            return True

            return False

    def clear_grad(self):

        with th.no_grad():

            for lane in self.lane.values():

                d_lane: FastMicroLane = lane

                d_lane.accel_max = d_lane.accel_max.detach()
                d_lane.accel_pref = d_lane.accel_pref.detach()
                d_lane.target_vel = d_lane.target_vel.detach()
                d_lane.time_pref = d_lane.time_pref.detach()
                d_lane.vehicle_length = d_lane.vehicle_length.detach()
                d_lane.min_space = d_lane.min_space.detach()

                d_lane.curr_pos = d_lane.curr_pos.detach()
                d_lane.curr_vel = d_lane.curr_vel.detach()
                d_lane.next_pos = d_lane.next_pos.detach()
                d_lane.next_vel = d_lane.next_vel.detach()

            self.idm_vehicle_position = self.idm_vehicle_position.detach()
            self.idm_vehicle_velocity = self.idm_vehicle_velocity.detach()
            self.idm_vehicle_heading = self.idm_vehicle_heading.detach()

            self.auto_vehicle_position = self.auto_vehicle_position.detach()
            self.auto_vehicle_velocity = self.auto_vehicle_velocity.detach()
            self.auto_vehicle_speed = self.auto_vehicle_speed.detach()
            self.auto_vehicle_diagonal = self.auto_vehicle_diagonal.detach()
            self.auto_vehicle_heading = self.auto_vehicle_heading.detach()
            self.auto_vehicle_length = self.auto_vehicle_length.detach()
            self.auto_vehicle_lane_distance = self.auto_vehicle_lane_distance.detach()
            self.auto_vehicle_lane_lateral = self.auto_vehicle_lane_lateral.detach()
            self.auto_vehicle_lane_longitudinal = self.auto_vehicle_lane_longitudinal.detach()

    def add_vehicles_to_road_for_view(self):

        with th.no_grad():

            self.hw_road.vehicles.clear()

            # idm vehicles;
            for i in range(self.num_idm_vehicle):
                pos = self.idm_vehicle_position[i].cpu().numpy()
                heading = self.idm_vehicle_heading[i].cpu().numpy()
                rv = Vehicle(self.hw_road, pos, heading)

                self.hw_road.vehicles.append(rv)
            
            # auto vehicles;
            observer_vehicle = None
            for i in range(self.num_auto_vehicle):
                pos = self.auto_vehicle_position[i].cpu().numpy()
                heading = self.auto_vehicle_heading[i].cpu().numpy()
                rv = Vehicle(self.hw_road, pos, heading)

                self.hw_road.vehicles.append(rv)

                if observer_vehicle == None:
                    observer_vehicle = rv

            return observer_vehicle

    def tensorize_value(self, value: np.ndarray, dtype=th.float32):
        return th.tensor(value, dtype=dtype, device=self.device)