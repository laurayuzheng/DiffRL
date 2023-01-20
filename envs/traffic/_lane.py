'''
@ author: SonSang (Sanghyun Son)
@ email: shh1295@gmail.com
'''
import torch as th

from externals.traffic.road.lane._base_lane import BaseLane
from externals.traffic.road.vehicle.micro_vehicle import MicroVehicle

def make_tensor(val, dtype, device):
    if isinstance(val, th.Tensor):
        return val
    else:
        return th.tensor([val], dtype=dtype, device=device)

class FastMicroLane(BaseLane):
    '''
    Here we assume i-th vehicle is right behind (i + 1)-th vehicle.
    '''            
    def __init__(self, lane_length: float, speed_limit: float, device):
        super().__init__(-1, lane_length, speed_limit)

        self.device = device

        # initialize vehicle state;
        self.vehicle_id: th.Tensor = th.zeros((0,), dtype=th.int32, device=device)
        self.curr_pos: th.Tensor = th.zeros((0,), dtype=th.float32, device=device)
        self.curr_vel: th.Tensor = th.zeros((0,), dtype=th.float32, device=device)
        self.accel_max: th.Tensor = th.zeros((0,), dtype=th.float32, device=device)
        self.accel_pref: th.Tensor = th.zeros((0,), dtype=th.float32, device=device)
        self.target_vel: th.Tensor = th.zeros((0,), dtype=th.float32, device=device)
        self.min_space: th.Tensor = th.zeros((0,), dtype=th.float32, device=device)
        self.time_pref: th.Tensor = th.zeros((0,), dtype=th.float32, device=device)
        self.vehicle_length: th.Tensor = th.zeros((0,), dtype=th.float32, device=device)

        # next states;
        self.next_pos: th.Tensor = th.zeros((0,), dtype=th.float32, device=device)
        self.next_vel: th.Tensor = th.zeros((0,), dtype=th.float32, device=device)

    def is_macro(self):
        return False

    def is_micro(self):
        return True

    def to(self, device, dtype):
        self.vehicle_id = self.vehicle_id.to(device=device, dtype=th.int32)
        self.curr_pos = self.curr_pos.to(device=device, dtype=dtype)
        self.curr_vel = self.curr_vel.to(device=device, dtype=dtype)
        self.accel_max = self.accel_max.to(device=device, dtype=dtype)
        self.accel_pref = self.accel_pref.to(device=device, dtype=dtype)
        self.target_vel = self.target_vel.to(device=device, dtype=dtype)
        self.min_space = self.min_space.to(device=device, dtype=dtype)
        self.time_pref = self.time_pref.to(device=device, dtype=dtype)
        self.vehicle_length = self.vehicle_length.to(device=device, dtype=dtype)

        self.next_pos = self.next_pos.to(device=device, dtype=dtype)
        self.next_vel = self.next_vel.to(device=device, dtype=dtype)

    def add_vehicle(self, nv: MicroVehicle):

        id = make_tensor(nv.id, th.int32, self.device)
        pos = make_tensor(nv.position, th.float32, self.device)
        vel = make_tensor(nv.speed, th.float32, self.device)
        accel_max = make_tensor(nv.accel_max, th.float32, self.device)
        accel_pref = make_tensor(nv.accel_pref, th.float32, self.device)
        target_vel = make_tensor(nv.target_speed, th.float32, self.device)
        min_space = make_tensor(nv.min_space, th.float32, self.device)
        time_pref = make_tensor(nv.time_pref, th.float32, self.device)
        vehicle_length = make_tensor(nv.length, th.float32, self.device)

        self.add_vehicle_tensor(id, pos, vel, accel_max, accel_pref, target_vel, min_space, time_pref, vehicle_length)

    def add_vehicle_tensor(self, 
                            id: th.Tensor,
                            pos: th.Tensor, 
                            vel: th.Tensor, 
                            accel_max: th.Tensor,
                            accel_pref: th.Tensor,
                            target_vel: th.Tensor,
                            min_space: th.Tensor,
                            time_pref: th.Tensor,
                            vehicle_length: th.Tensor):

        # find place to put this vehicle;

        assert pos >= 0 and pos <= self.length, ""

        if self.num_vehicle() == 0:
            self.vehicle_id = id
            self.curr_pos = pos
            self.curr_vel = vel
            self.accel_max = accel_max
            self.accel_pref = accel_pref
            self.target_vel = target_vel
            self.min_space = min_space
            self.time_pref = time_pref
            self.vehicle_length = vehicle_length

        else:

            idx = -1

            for i in range(self.num_vehicle()):

                n_pos = self.curr_pos[i]
                if n_pos > pos:
                    idx = i
                    break

            self.vehicle_id = th.cat([self.vehicle_id[:idx], id, self.vehicle_id[idx:]])
            self.curr_pos = th.cat([self.curr_pos[:idx], pos, self.curr_pos[idx:]])
            self.curr_vel = th.cat([self.curr_vel[:idx], vel, self.curr_vel[idx:]])
            self.accel_max = th.cat([self.accel_max[:idx], accel_max, self.accel_max[idx:]])
            self.accel_pref = th.cat([self.accel_pref[:idx], accel_pref, self.accel_pref[idx:]])
            self.target_vel = th.cat([self.target_vel[:idx], target_vel, self.target_vel[idx:]])
            self.min_space = th.cat([self.min_space[:idx], min_space, self.min_space[idx:]])
            self.time_pref = th.cat([self.time_pref[:idx], time_pref, self.time_pref[idx:]])
            self.vehicle_length = th.cat([self.vehicle_length[:idx], vehicle_length, self.vehicle_length[idx:]])

    def add_tail_vehicle_tensor(self, 
                                id: th.Tensor,
                                pos: th.Tensor, 
                                vel: th.Tensor, 
                                accel_max: th.Tensor,
                                accel_pref: th.Tensor,
                                target_vel: th.Tensor,
                                min_space: th.Tensor,
                                time_pref: th.Tensor,
                                vehicle_length: th.Tensor):

        '''
        Add a vehicle at the end of the lane.
        '''

        self.vehicle_id = th.cat((id, self.vehicle_id), dim=0)
        self.curr_pos = th.cat((pos, self.curr_pos), dim=0)
        self.curr_vel = th.cat((vel, self.curr_vel), dim=0)
        self.next_pos = th.cat((pos, self.next_pos), dim=0)
        self.next_vel = th.cat((vel, self.next_vel), dim=0)
        self.accel_max = th.cat((accel_max, self.accel_max), dim=0)
        self.accel_pref = th.cat((accel_pref, self.accel_pref), dim=0)
        self.target_vel = th.cat((target_vel, self.target_vel), dim=0)
        self.min_space = th.cat((min_space, self.min_space), dim=0)
        self.time_pref = th.cat((time_pref, self.time_pref), dim=0)
        self.vehicle_length = th.cat((vehicle_length, self.vehicle_length), dim=0)
        
    def remove_head_vehicle(self):
        self.vehicle_id = self.vehicle_id[:-1]
        self.curr_pos = self.curr_pos[:-1]
        self.curr_vel = self.curr_vel[:-1]
        self.next_pos = self.next_pos[:-1]
        self.next_vel = self.next_vel[:-1]
        self.accel_max = self.accel_max[:-1]
        self.accel_pref = self.accel_pref[:-1]
        self.target_vel = self.target_vel[:-1]
        self.min_space = self.min_space[:-1]
        self.time_pref = self.time_pref[:-1]
        self.vehicle_length = self.vehicle_length[:-1]

    def get_head_vehicle_id(self):

        if self.num_vehicle() == 0:
            return th.tensor([-1], device=self.device)

        return self.vehicle_id[-1]
        
    def preprocess(self, 
                    last_pos_delta: th.Tensor, 
                    last_vel_delta: th.Tensor, 
                    delta_time: float):
        '''
        For given vehicle information, generate a set of tensors that can be fed into IDMLayer.
        '''
        
        num_vehicle = self.num_vehicle()

        # pos_delta: assume position[i + 1] > position[i];
        pos_delta = th.zeros_like(self.curr_pos)
        pos_delta[:num_vehicle-1] = self.curr_pos[1:] - self.curr_pos[:num_vehicle-1] - \
                                        ((self.vehicle_length[1:] + self.vehicle_length[:num_vehicle-1]) * 0.5)
        pos_delta[num_vehicle-1] = last_pos_delta

        overlap_indices = th.where(pos_delta < 0.)[0]
        if len(overlap_indices) > 0:
            print("Detected IDM vehicles overlap (# = {})".format(len(overlap_indices)))
        
        pos_delta = th.clamp(pos_delta, min=0.)
        
        # vel_delta;
        vel_delta = th.zeros_like(self.curr_vel)
        vel_delta[:num_vehicle-1] = self.curr_vel[:num_vehicle - 1] - self.curr_vel[1:]
        vel_delta[num_vehicle-1] = last_vel_delta

        # delta time;
        delta_time = th.ones_like(self.curr_pos) * delta_time

        return pos_delta, vel_delta, delta_time

    def num_vehicle(self):
        return self.curr_pos.shape[0]

    def update_state(self):
        '''
        update current state with next state
        '''
        self.curr_pos = self.next_pos
        self.curr_vel = self.next_vel

    def print(self):
        print("Micro Lane: # vehicle = {} / lane length = {:.2f} m / speed_limit = {:.2f} m/sec".format(
            self.num_vehicle(),
            self.length,
            self.speed_limit,
        ))

        print("pos: {}".format(list(self.curr_pos.detach().cpu().numpy()[:, 0])))
        print("vel: {}".format(list(self.curr_vel.detach().cpu().numpy()[:, 0])))