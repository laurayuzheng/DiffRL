import torch as th

def vehicle_lane_velocity(vehicle_speed: th.Tensor,
                            vehicle_lane_heading: th.Tensor):

    assert vehicle_speed.ndim == 1, ""
    assert vehicle_lane_heading.ndim == 2, ""
    assert vehicle_speed.shape[0] == vehicle_lane_heading.shape[0], ""

    # dim = [num vehicle, num lane, 2]
    t_vehicle_direction_x = th.cos(vehicle_lane_heading)
    t_vehicle_direction_y = th.sin(vehicle_lane_heading)
    t_vehicle_direction = th.stack([t_vehicle_direction_x, t_vehicle_direction_y], dim=-1)

    t_vehicle_speed = vehicle_speed.unsqueeze(-1).unsqueeze(-1)

    velocity = t_vehicle_direction * t_vehicle_speed
    
    return velocity

def vehicle_lane_distance(vehicle_lane_longitudinal: th.Tensor,
                            vehicle_lane_lateral: th.Tensor,
                            lane_length: th.Tensor):

    assert vehicle_lane_longitudinal.ndim == 2, ""
    assert vehicle_lane_lateral.ndim == 2, ""
    assert lane_length.ndim == 1, ""

    num_vehicle = vehicle_lane_longitudinal.shape[0]

    t_lane_length = lane_length.unsqueeze(0).expand((num_vehicle, -1))

    dist = th.abs(vehicle_lane_lateral) + \
        th.clamp(vehicle_lane_longitudinal - t_lane_length, min=0) + \
        th.clamp(0 - vehicle_lane_longitudinal, min=0)

    return dist

'''
STRAIGHT LANE ================================================
'''
def straight_lane_position(vehicle_longitudinal: th.Tensor,
                            vehicle_lateral: th.Tensor,
                            lane_start: th.Tensor,
                            lane_direction: th.Tensor,
                            lane_direction_lateral: th.Tensor):

    assert vehicle_longitudinal.ndim == 1, ""
    assert vehicle_lateral.ndim == 1, ""
    assert lane_start.ndim == 2, ""
    assert lane_direction.ndim == 2, ""
    assert lane_direction_lateral.ndim == 2, ""

    num_vehicle = len(vehicle_longitudinal)
    num_lane = len(lane_start)

    # dim = [num_vehicle, num_lane, #]
    t_vehicle_longitudinal = vehicle_longitudinal.unsqueeze(-1).expand([-1, num_lane]).unsqueeze(-1)
    t_vehicle_lateral = vehicle_lateral.unsqueeze(-1).expand([-1, num_lane]).unsqueeze(-1)
    
    t_lane_start = lane_start.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_direction = lane_direction.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_direction_lateral = lane_direction_lateral.unsqueeze(0).expand([num_vehicle, -1, -1])

    position = t_lane_start + t_vehicle_longitudinal * t_lane_direction + t_vehicle_lateral * t_lane_direction_lateral
    
    return position

def straight_lane_heading(vehicle_longitudinal: th.Tensor, 
                                    vehicle_lateral: th.Tensor,
                                    vehicle_speed: th.Tensor, 
                                    lane_start: th.Tensor,
                                    lane_end: th.Tensor,
                                    lane_heading: th.Tensor,
                                    lane_direction: th.Tensor,
                                    lane_direction_lateral: th.Tensor):
    assert lane_heading.ndim == 1, ""

    num_vehicle = len(vehicle_longitudinal)
    num_lane = len(lane_start)

    # dim = [num_vehicle, num_lane, #]
    t_lane_heading = lane_heading.unsqueeze(0).expand([num_vehicle, -1])
    
    return t_lane_heading

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

    position = straight_lane_position(vehicle_longitudinal,
                                            vehicle_lateral,
                                            lane_start,
                                            lane_direction,
                                            lane_direction_lateral)

    heading = straight_lane_heading(vehicle_longitudinal, 
                                    vehicle_lateral,
                                    vehicle_speed, 
                                    lane_start,
                                    lane_end,
                                    lane_heading,
                                    lane_direction,
                                    lane_direction_lateral)

    velocity = vehicle_lane_velocity(vehicle_speed, heading)
    
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

    dist = th.abs(lateral) + th.clamp(longitudinal - t_lane_length, min=0) + th.clamp(0 - longitudinal, min=0)

    return dist, longitudinal, lateral

'''
SINE LANE =========================================
'''
def sine_lane_position(vehicle_longitudinal: th.Tensor, 
                                    vehicle_lateral: th.Tensor,
                                    vehicle_speed: th.Tensor, 
                                    lane_start: th.Tensor,
                                    lane_end: th.Tensor,
                                    lane_heading: th.Tensor,
                                    lane_direction: th.Tensor,
                                    lane_direction_lateral: th.Tensor,
                                    lane_amplitude: th.Tensor,
                                    lane_pulsation: th.Tensor,
                                    lane_phase: th.Tensor):
    
    assert vehicle_longitudinal.ndim == 1, ""
    assert vehicle_lateral.ndim == 1, ""
    assert vehicle_speed.ndim == 1, ""
    assert lane_start.ndim == 2, ""
    assert lane_end.ndim == 2, ""
    assert lane_heading.ndim == 1, ""
    assert lane_direction.ndim == 2, ""
    assert lane_direction_lateral.ndim == 2, ""
    assert lane_amplitude.ndim == 1, ""
    assert lane_pulsation.ndim == 1, ""
    assert lane_phase.ndim == 1, ""

    num_vehicle = len(vehicle_longitudinal)
    num_lane = len(lane_start)

    # dim = [num_vehicle, num_lane, #]
    t_vehicle_longitudinal = vehicle_longitudinal.unsqueeze(-1).expand((-1, num_lane))
    t_vehicle_lateral = vehicle_lateral.unsqueeze(-1).expand((-1, num_lane))
    t_lane_start = lane_start.unsqueeze(0).expand((num_vehicle, -1, -1))
    t_lane_direction = lane_direction.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_direction_lateral = lane_direction_lateral.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_amplitude = lane_amplitude.unsqueeze(0).expand((num_vehicle, -1))
    t_lane_pulsation = lane_pulsation.unsqueeze(0).expand((num_vehicle, -1))
    t_lane_phase = lane_phase.unsqueeze(0).expand((num_vehicle, -1))
    
    n_vehicle_longitudinal = t_vehicle_longitudinal
    n_vehicle_lateral = t_vehicle_lateral + t_lane_amplitude * th.sin(t_lane_pulsation * t_vehicle_longitudinal + t_lane_phase)

    n_vehicle_longitudinal = n_vehicle_longitudinal.unsqueeze(-1)
    n_vehicle_lateral = n_vehicle_lateral.unsqueeze(-1)

    position = t_lane_start + n_vehicle_longitudinal * t_lane_direction + n_vehicle_lateral * t_lane_direction_lateral
    
    return position

def sine_lane_heading(vehicle_longitudinal: th.Tensor, 
                        vehicle_lateral: th.Tensor,
                        vehicle_speed: th.Tensor, 
                        lane_start: th.Tensor,
                        lane_end: th.Tensor,
                        lane_heading: th.Tensor,
                        lane_direction: th.Tensor,
                        lane_direction_lateral: th.Tensor,
                        lane_amplitude: th.Tensor,
                        lane_pulsation: th.Tensor,
                        lane_phase: th.Tensor):

    assert vehicle_longitudinal.ndim == 1, ""
    assert vehicle_lateral.ndim == 1, ""
    assert vehicle_speed.ndim == 1, ""
    assert lane_start.ndim == 2, ""
    assert lane_end.ndim == 2, ""
    assert lane_heading.ndim == 1, ""
    assert lane_direction.ndim == 2, ""
    assert lane_direction_lateral.ndim == 2, ""
    assert lane_amplitude.ndim == 1, ""
    assert lane_pulsation.ndim == 1, ""
    assert lane_phase.ndim == 1, ""

    num_vehicle = len(vehicle_longitudinal)
    num_lane = len(lane_start)

    # dim = [num_vehicle, num_lane, #]
    t_vehicle_longitudinal = vehicle_longitudinal.unsqueeze(-1).expand((-1, num_lane))
    t_vehicle_lateral = vehicle_lateral.unsqueeze(-1).expand((-1, num_lane))
    t_lane_start = lane_start.unsqueeze(0).expand((num_vehicle, -1, -1))
    t_lane_heading = lane_heading.unsqueeze(0).expand((num_vehicle, -1))
    t_lane_direction = lane_direction.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_direction_lateral = lane_direction_lateral.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_amplitude = lane_amplitude.unsqueeze(0).expand((num_vehicle, -1))
    t_lane_pulsation = lane_pulsation.unsqueeze(0).expand((num_vehicle, -1))
    t_lane_phase = lane_phase.unsqueeze(0).expand((num_vehicle, -1))
    
    heading = t_lane_heading + th.arctan(t_lane_amplitude * t_lane_pulsation * \
        th.cos(t_lane_pulsation * t_vehicle_longitudinal + t_lane_phase))

    return heading
    

def sine_lane_position_velocity(vehicle_longitudinal: th.Tensor, 
                                    vehicle_lateral: th.Tensor,
                                    vehicle_speed: th.Tensor, 
                                    lane_start: th.Tensor,
                                    lane_end: th.Tensor,
                                    lane_heading: th.Tensor,
                                    lane_direction: th.Tensor,
                                    lane_direction_lateral: th.Tensor,
                                    lane_amplitude: th.Tensor,
                                    lane_pulsation: th.Tensor,
                                    lane_phase: th.Tensor):

    assert vehicle_longitudinal.ndim == 1, ""
    assert vehicle_lateral.ndim == 1, ""
    assert vehicle_speed.ndim == 1, ""
    assert lane_start.ndim == 2, ""
    assert lane_end.ndim == 2, ""
    assert lane_heading.ndim == 1, ""
    assert lane_direction.ndim == 2, ""
    assert lane_direction_lateral.ndim == 2, ""
    assert lane_amplitude.ndim == 1, ""
    assert lane_pulsation.ndim == 1, ""
    assert lane_phase.ndim == 1, ""

    position = sine_lane_position(vehicle_longitudinal,
                                    vehicle_lateral,
                                    vehicle_speed,
                                    lane_start,
                                    lane_end,
                                    lane_heading,
                                    lane_direction,
                                    lane_direction_lateral,
                                    lane_amplitude,
                                    lane_pulsation,
                                    lane_phase)

    heading = sine_lane_heading(vehicle_longitudinal,
                                    vehicle_lateral,
                                    vehicle_speed,
                                    lane_start,
                                    lane_end,
                                    lane_heading,
                                    lane_direction,
                                    lane_direction_lateral,
                                    lane_amplitude,
                                    lane_pulsation,
                                    lane_phase)

    velocity = vehicle_lane_velocity(vehicle_speed, heading)
    
    return position, velocity, heading

def sine_lane_local_coords(vehicle_position: th.Tensor, 
                            lane_start: th.Tensor,
                            lane_end: th.Tensor,
                            lane_heading: th.Tensor,
                            lane_direction: th.Tensor,
                            lane_direction_lateral: th.Tensor,
                            lane_amplitude: th.Tensor,
                            lane_pulsation: th.Tensor,
                            lane_phase: th.Tensor):

    assert vehicle_position.ndim == 2, ""
    assert lane_start.ndim == 2, ""
    assert lane_end.ndim == 2, ""
    assert lane_heading.ndim == 1, ""
    assert lane_direction.ndim == 2, ""
    assert lane_direction_lateral.ndim == 2, ""
    assert lane_amplitude.ndim == 1, ""
    assert lane_pulsation.ndim == 1, ""
    assert lane_phase.ndim == 1, ""

    longitudinal, lateral = straight_lane_local_coords(vehicle_position, 
                                                        lane_start,
                                                        lane_end,
                                                        lane_heading,
                                                        lane_direction,
                                                        lane_direction_lateral)

    num_vehicle = len(vehicle_position)

    t_lane_amplitude = lane_amplitude.unsqueeze(0).expand((num_vehicle, -1))
    t_lane_pulsation = lane_pulsation.unsqueeze(0).expand((num_vehicle, -1))
    t_lane_phase = lane_phase.unsqueeze(0).expand((num_vehicle, -1))
    
    n_longitudinal = longitudinal
    n_lateral = lateral - t_lane_amplitude * th.sin(t_lane_pulsation * longitudinal + t_lane_phase)

    return n_longitudinal, n_lateral

def sine_lane_distance(vehicle_position: th.Tensor, 
                            lane_start: th.Tensor,
                            lane_end: th.Tensor,
                            lane_heading: th.Tensor,
                            lane_direction: th.Tensor,
                            lane_direction_lateral: th.Tensor,
                            lane_amplitude: th.Tensor,
                            lane_pulsation: th.Tensor,
                            lane_phase: th.Tensor):

    assert vehicle_position.ndim == 2, ""
    assert lane_start.ndim == 2, ""
    assert lane_end.ndim == 2, ""
    assert lane_heading.ndim == 1, ""
    assert lane_direction.ndim == 2, ""
    assert lane_direction_lateral.ndim == 2, ""
    assert lane_amplitude.ndim == 1, ""
    assert lane_pulsation.ndim == 1, ""
    assert lane_phase.ndim == 1, ""

    longitudinal, lateral = sine_lane_local_coords(vehicle_position,
                                                        lane_start,
                                                        lane_end,
                                                        lane_heading,
                                                        lane_direction,
                                                        lane_direction_lateral,
                                                        lane_amplitude,
                                                        lane_pulsation,
                                                        lane_phase)
                                                        
    num_vehicle = len(vehicle_position)
    num_lane = len(lane_start)

    t_lane_start = lane_start.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_end = lane_end.unsqueeze(0).expand([num_vehicle, -1, -1])
    t_lane_offset = t_lane_end - t_lane_start
    t_lane_length = th.norm(t_lane_offset, p=2, dim=2)

    dist = th.abs(lateral) + th.clamp(longitudinal - t_lane_length, min=0) + th.clamp(0 - longitudinal, min=0)

    return dist, longitudinal, lateral

'''
CIRCULAR LANE =========================================
'''

def circular_lane_position(vehicle_longitudinal: th.Tensor, 
                            vehicle_lateral: th.Tensor,
                            vehicle_speed: th.Tensor, 
                            lane_center: th.Tensor,
                            lane_radius: th.Tensor,
                            lane_start_phase: th.Tensor,
                            lane_end_phase: th.Tensor,
                            lane_clockwise: th.Tensor):
    
    assert vehicle_longitudinal.ndim == 1, ""
    assert vehicle_lateral.ndim == 1, ""
    assert vehicle_speed.ndim == 1, ""
    assert lane_center.ndim == 2, ""
    assert lane_radius.ndim == 1, ""
    assert lane_start_phase.ndim == 1, ""
    assert lane_end_phase.ndim == 1, ""
    assert lane_clockwise.ndim == 1, ""

    num_vehicle = len(vehicle_longitudinal)
    num_lane = len(lane_center)

    # dim = [num_vehicle, num_lane, #]
    t_vehicle_longitudinal = vehicle_longitudinal.unsqueeze(-1).expand((-1, num_lane))
    t_vehicle_lateral = vehicle_lateral.unsqueeze(-1).expand((-1, num_lane))
    t_lane_center = lane_center.unsqueeze(0).expand((num_vehicle, -1, -1))
    t_lane_direction = lane_clockwise.unsqueeze(0).expand((num_vehicle, num_lane))
    t_lane_direction = th.where(t_lane_direction == 1, \
                        th.ones_like(t_lane_direction, dtype=th.float32), \
                        th.ones_like(t_lane_direction, dtype=th.float32) * -1.)
    t_lane_radius = lane_radius.unsqueeze(0).expand((num_vehicle, num_lane)).clamp(min=1e-3)
    t_lane_start_phase = lane_start_phase.unsqueeze(0).expand((num_vehicle, num_lane))

    t_phi = t_lane_direction * t_vehicle_longitudinal / t_lane_radius + t_lane_start_phase
    t_phi_cx = th.cos(t_phi)
    t_phi_cy = th.sin(t_phi)
    t_phi_c = th.stack([t_phi_cx, t_phi_cy], -1)
    position = t_lane_center + (t_lane_radius - t_vehicle_lateral * t_lane_direction).unsqueeze(-1) * t_phi_c

    return position

def circular_lane_heading(vehicle_longitudinal: th.Tensor, 
                            vehicle_lateral: th.Tensor,
                            vehicle_speed: th.Tensor, 
                            lane_center: th.Tensor,
                            lane_radius: th.Tensor,
                            lane_start_phase: th.Tensor,
                            lane_end_phase: th.Tensor,
                            lane_clockwise: th.Tensor):
    
    assert vehicle_longitudinal.ndim == 1, ""
    assert vehicle_lateral.ndim == 1, ""
    assert vehicle_speed.ndim == 1, ""
    assert lane_center.ndim == 2, ""
    assert lane_radius.ndim == 1, ""
    assert lane_start_phase.ndim == 1, ""
    assert lane_end_phase.ndim == 1, ""
    assert lane_clockwise.ndim == 1, ""

    num_vehicle = len(vehicle_longitudinal)
    num_lane = len(lane_center)

    # dim = [num_vehicle, num_lane, #]
    t_vehicle_longitudinal = vehicle_longitudinal.unsqueeze(-1).expand((-1, num_lane))
    t_vehicle_lateral = vehicle_lateral.unsqueeze(-1).expand((-1, num_lane))
    t_lane_center = lane_center.unsqueeze(0).expand((num_vehicle, -1, -1))
    t_lane_direction = lane_clockwise.unsqueeze(0).expand((num_vehicle, num_lane))
    t_lane_direction = th.where(t_lane_direction == 1, \
                        th.ones_like(t_lane_direction, dtype=th.float32), \
                        th.ones_like(t_lane_direction, dtype=th.float32) * -1.)
    t_lane_radius = lane_radius.unsqueeze(0).expand((num_vehicle, num_lane)).clamp(min=1e-3)
    t_lane_start_phase = lane_start_phase.unsqueeze(0).expand((num_vehicle, num_lane))

    t_phi = t_lane_direction * t_vehicle_longitudinal / t_lane_radius + t_lane_start_phase
    t_psi = t_phi + th.pi / 2 * t_lane_direction

    return t_psi


def circular_lane_position_velocity(vehicle_longitudinal: th.Tensor, 
                            vehicle_lateral: th.Tensor,
                            vehicle_speed: th.Tensor, 
                            lane_center: th.Tensor,
                            lane_radius: th.Tensor,
                            lane_start_phase: th.Tensor,
                            lane_end_phase: th.Tensor,
                            lane_clockwise: th.Tensor):
    
    assert vehicle_longitudinal.ndim == 1, ""
    assert vehicle_lateral.ndim == 1, ""
    assert vehicle_speed.ndim == 1, ""
    assert lane_center.ndim == 2, ""
    assert lane_radius.ndim == 1, ""
    assert lane_start_phase.ndim == 1, ""
    assert lane_end_phase.ndim == 1, ""
    assert lane_clockwise.ndim == 1, ""

    position = circular_lane_position(vehicle_longitudinal,
                                        vehicle_lateral,
                                        vehicle_speed,
                                        lane_center,
                                        lane_radius,
                                        lane_start_phase,
                                        lane_end_phase,
                                        lane_clockwise)

    heading = circular_lane_heading(vehicle_longitudinal,
                                        vehicle_lateral,
                                        vehicle_speed,
                                        lane_center,
                                        lane_radius,
                                        lane_start_phase,
                                        lane_end_phase,
                                        lane_clockwise)

    velocity = vehicle_lane_velocity(vehicle_speed, heading)
    
    return position, velocity, heading

def circular_lane_local_coords(vehicle_position: th.Tensor, 
                            lane_center: th.Tensor,
                            lane_radius: th.Tensor,
                            lane_start_phase: th.Tensor,
                            lane_end_phase: th.Tensor,
                            lane_clockwise: th.Tensor):
    
    assert vehicle_position.ndim == 2, ""
    assert lane_center.ndim == 2, ""
    assert lane_radius.ndim == 1, ""
    assert lane_start_phase.ndim == 1, ""
    assert lane_end_phase.ndim == 1, ""
    assert lane_clockwise.ndim == 1, ""

    num_vehicle = len(vehicle_position)
    num_lane = len(lane_center)

    # dim = [num_vehicle, num_lane, #]

    t_vehicle_position = vehicle_position.unsqueeze(-1).expand((-1, num_lane, -1))
    t_lane_center = lane_center.unsqueeze(0).expand((num_vehicle, -1, -1))
    t_lane_direction = lane_clockwise.unsqueeze(0).expand((num_vehicle, num_lane))
    t_lane_direction = th.where(t_lane_direction == 1, \
                        th.ones_like(t_lane_direction, dtype=th.float32), \
                        th.ones_like(t_lane_direction, dtype=th.float32) * -1.)
    t_lane_radius = lane_radius.unsqueeze(0).expand((num_vehicle, num_lane))
    t_lane_start_phase = lane_start_phase.unsqueeze(0).expand((num_vehicle, num_lane))

    delta = t_vehicle_position - t_lane_center
    phi = th.arctan2(delta[:, :, 1], delta[:, :, 0].clamp(min=1e-3))
    w_phi = (((phi - t_lane_start_phase) + th.pi) % (2 * th.pi)) - th.pi
    phi = t_lane_start_phase + w_phi
    r = th.norm(delta, p=2, dim=2)

    longitudinal = t_lane_direction * (phi - t_lane_start_phase) * t_lane_radius
    lateral = t_lane_direction * (t_lane_radius - r)

    return longitudinal, lateral

def circular_lane_distance(vehicle_position: th.Tensor, 
                            lane_center: th.Tensor,
                            lane_radius: th.Tensor,
                            lane_start_phase: th.Tensor,
                            lane_end_phase: th.Tensor,
                            lane_clockwise: th.Tensor):
    
    assert vehicle_position.ndim == 2, ""
    assert lane_center.ndim == 2, ""
    assert lane_radius.ndim == 1, ""
    assert lane_start_phase.ndim == 1, ""
    assert lane_end_phase.ndim == 1, ""
    assert lane_clockwise.ndim == 1, ""

    longitudinal, lateral = circular_lane_local_coords(vehicle_position,
                                                        lane_center,
                                                        lane_radius,
                                                        lane_start_phase,
                                                        lane_end_phase,
                                                        lane_clockwise)

                                                        
    num_vehicle = len(vehicle_position)
    num_lane = len(lane_center)

    t_lane_direction = lane_clockwise.unsqueeze(0).expand((num_vehicle, num_lane))
    t_lane_direction = th.where(t_lane_direction == 1, \
                        th.ones_like(t_lane_direction, dtype=th.float32), \
                        th.ones_like(t_lane_direction, dtype=th.float32) * -1.)
    t_lane_radius = lane_radius.unsqueeze(0).expand((num_vehicle, num_lane))
    t_lane_start_phase = lane_start_phase.unsqueeze(0).expand((num_vehicle, num_lane))
    t_lane_end_phase = lane_end_phase.unsqueeze(0).expand((num_vehicle, num_lane))

    t_lane_length = t_lane_radius * (t_lane_end_phase - t_lane_start_phase) * t_lane_direction

    dist = th.abs(lateral) + th.clamp(longitudinal - t_lane_length, min=0) + th.clamp(0 - longitudinal, min=0)

    return dist, longitudinal, lateral