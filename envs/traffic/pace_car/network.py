import numpy as np
from envs.traffic._network import ParallelRoadNetwork, LineType, AbstractLane
from envs.traffic.diff_highway_env.kinematics import Vehicle

import torch as th

class PaceCarRoadNetwork(ParallelRoadNetwork):

    '''
    Specialized simulator for Pace car problem.

    In this environment, there are multiple (or single) parallel lanes where
    IDM vehicles run on. In there, our auto vehicles should maneuver to restrict
    the vehicles speed as given as an input.
    '''

    def __init__(self, 
                    speed_limit: float, 
                    desired_speed_limit: float, 
                    num_auto_vehicle: int, 
                    num_idm_vehicle: int, 
                    num_all_lane: int,
                    device):

        super().__init__(speed_limit, num_idm_vehicle, num_auto_vehicle, num_all_lane, device)

        '''
        In this problem, [desired_speed_limit] is the very speed limit that
        the auto vehicles should enforce the IDM vehicles. So usually it should 
        be smaller than the lane's speed limit.
        '''
        self.desired_speed_limit = desired_speed_limit

        self.lane_length = 1e6      # long enough lanes;

        self.reset()

    def reset(self):

        super().reset()

        # allocate idm vehicles;
        num_idm_vehicle_per_lane = []
        for i in range(self.num_all_lane):
            num_idm_vehicle_per_lane.append(self.num_idm_vehicle // self.num_all_lane)

        num_remain_idm_vehicle = self.num_idm_vehicle - (self.num_idm_vehicle // self.num_all_lane) * self.num_all_lane
        for i in range(self.num_all_lane):
            if num_remain_idm_vehicle == 0:
                break
            num_idm_vehicle_per_lane[i] += 1
            num_remain_idm_vehicle -= 1

        # allocate auto vehicles;
        num_auto_vehicle_per_lane = []
        for i in range(self.num_all_lane):
            num_auto_vehicle_per_lane.append(self.num_auto_vehicle // self.num_all_lane)

        num_remain_auto_vehicle = self.num_auto_vehicle - (self.num_auto_vehicle // self.num_all_lane) * self.num_all_lane
        for i in range(self.num_all_lane):
            if num_remain_auto_vehicle == 0:
                break
            num_auto_vehicle_per_lane[i] += 1
            num_remain_auto_vehicle -= 1

        # make lanes;
        lane_width = AbstractLane.DEFAULT_WIDTH
        for i in range(self.num_all_lane):
            start = np.array([0, float(i) * lane_width])
            end = np.array([self.lane_length, float(i) * lane_width])
            line_type = [LineType.STRIPED, LineType.STRIPED]
            if i == 0:
                line_type[0] = LineType.CONTINUOUS
            if i == self.num_lane - 1:
                line_type[1] = LineType.CONTINUOUS

            self.make_straight_lane(start, end, "start", "end", line_type)

        # make idm vehicles;
        lane_max_pos = []
        for i in range(self.num_all_lane):
            num_idm_vehicle_on_curr_lane = num_idm_vehicle_per_lane[i]

            curr_lane_max_pos = 0
            for j in range(num_idm_vehicle_on_curr_lane):
                nv, nr = self.create_default_vehicle_with_random_route(i)
                nv.position = j * 2.0 * nv.length
                self.add_vehicle(nv, nr)

                curr_lane_max_pos = nv.position

            lane_max_pos.append(curr_lane_max_pos)

        # make auto vehicles;
        auto_vehicle_id = 0
        for i in range(self.num_all_lane):
            lane = self.comp_lane[i].env_lane
            curr_lane_max_pos = lane_max_pos[i]
            num_auto_vehicle_on_curr_lane = num_auto_vehicle_per_lane[i]

            for j in range(num_auto_vehicle_on_curr_lane):
                longitudinal = curr_lane_max_pos + 50.0 + 10.0 * j
                pos = lane.position(longitudinal, 0.0)
                heading = lane.heading_at(longitudinal)
                
                nv = Vehicle(self.hw_road, pos, heading, 0)
                self.add_auto_vehicle(auto_vehicle_id, nv)
                auto_vehicle_id += 1

        # update auto vehicle info for first turn;
        self.update_tensor_info()