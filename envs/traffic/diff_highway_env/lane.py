from highway_env.road.lane import StraightLane

import torch as th

def straight_lane_position_velocity(vehicle_longitudinal: th.Tensor, 
                                    vehicle_lateral: th.Tensor,
                                    vehicle_speed: th.Tensor, 
                                    lane_start: th.Tensor,
                                    lane_end: th.Tensor,
                                    lane_heading: th.Tensor,
                                    lane_direction: th.Tensor,
                                    lane_direction_lateral: th.Tensor):

    assert vehicle_longitudinal.ndim == 1, ""
    assert vehicle_lateral.ndim == 1, ""
    assert vehicle_speed.ndim == 1, ""
    assert lane_start.ndim == 2, ""
    assert lane_end.ndim == 2, ""
    assert lane_heading.ndim == 1, ""
    assert lane_direction.ndim == 2, ""
    assert lane_direction_lateral.ndim == 2, ""

    num_vehicle = len(vehicle_longitudinal)
    num_lane = len(lane_start)

    # dim = [num_vehicle, num_lane, #]
    t_vehicle_longitudinal = vehicle_longitudinal.unsqueeze(-1).expand([-1, num_lane]).unsqueeze(-1)
    t_vehicle_lateral = vehicle_lateral.unsqueeze(-1).expand([-1, num_lane]).unsqueeze(-1)
    t_vehicle_speed = vehicle_speed.unsqueeze(-1).expand([-1, num_lane]).unsqueeze(-1)

    t_lane_start = lane_start.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_end = lane_end.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_heading = lane_heading.unsqueeze(0).expand([num_vehicle, -1]).unsqueeze(-1)
    t_lane_direction = lane_direction.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_direction_lateral = lane_direction_lateral.unsqueeze(0).expand([num_vehicle, -1, -1])

    position = t_lane_start + t_vehicle_longitudinal * t_lane_direction + t_vehicle_lateral * t_lane_direction_lateral
    direction = t_lane_direction
    velocity = direction * t_vehicle_speed
    heading = t_lane_heading.squeeze(-1)

    return position, velocity, heading

def straight_lane_local_coords(vehicle_position: th.Tensor, 
                                lane_start: th.Tensor,
                                lane_end: th.Tensor,
                                lane_heading: th.Tensor,
                                lane_direction: th.Tensor,
                                lane_direction_lateral: th.Tensor):

    assert vehicle_position.ndim == 2, ""
    assert lane_start.ndim == 2, ""
    assert lane_end.ndim == 2, ""
    assert lane_heading.ndim == 1, ""
    assert lane_direction.ndim == 2, ""
    assert lane_direction_lateral.ndim == 2, ""

    num_vehicle = len(vehicle_position)
    num_lane = len(lane_start)

    # dim = [num_vehicle, num_lane, #]
    t_vehicle_position = vehicle_position.unsqueeze(1).expand([-1, num_lane, -1])

    t_lane_start = lane_start.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_end = lane_end.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_heading = lane_heading.unsqueeze(0).expand([num_vehicle, -1]).unsqueeze(-1)
    t_lane_direction = lane_direction.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_direction_lateral = lane_direction_lateral.unsqueeze(0).expand([num_vehicle, -1, -1])

    delta = t_vehicle_position - t_lane_start
        
    longitudinal = th.matmul(delta.unsqueeze(-2), t_lane_direction.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    lateral = th.matmul(delta.unsqueeze(-2), t_lane_direction_lateral.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    
    return longitudinal, lateral

def straight_lane_distance(vehicle_position: th.Tensor, 
                                lane_start: th.Tensor,
                                lane_end: th.Tensor,
                                lane_heading: th.Tensor,
                                lane_direction: th.Tensor,
                                lane_direction_lateral: th.Tensor):

    assert vehicle_position.ndim == 2, ""
    assert lane_start.ndim == 2, ""
    assert lane_end.ndim == 2, ""
    assert lane_heading.ndim == 1, ""
    assert lane_direction.ndim == 2, ""
    assert lane_direction_lateral.ndim == 2, ""

    longitudinal, lateral = straight_lane_local_coords(
                                vehicle_position, 
                                lane_start, 
                                lane_end, 
                                lane_heading, 
                                lane_direction, 
                                lane_direction_lateral)

    num_vehicle = len(vehicle_position)
    num_lane = len(lane_start)

    t_lane_start = lane_start.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_end = lane_end.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_offset = t_lane_end - t_lane_start
    t_lane_length = th.norm(t_lane_offset, p=2, dim=2)

    return th.abs(lateral) + th.clamp(longitudinal - t_lane_length, min=0) + th.clamp(0 - longitudinal, min=0)