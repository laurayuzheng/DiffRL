from highway_env.vehicle.kinematics import Vehicle, Road
import numpy as np

import torch as th

class dVehicle(Vehicle):

    def __init__(self,
                 road: Road,
                 position: th.Tensor,
                 heading: th.Tensor,
                 speed: th.Tensor):

        self.road = road
        self.position = position
        self.heading = heading
        self.speed = speed

        self.diagonal = np.sqrt(self.LENGTH**2 + self.WIDTH**2)
        self.action = {'steering': 0, 'acceleration': 0}
        
    def step(self, dt: float) -> None:
        """
        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        delta_f = self.action['steering']
        beta = th.arctan(1 / 2 * th.tan(delta_f))
        v = self.speed * th.tensor([th.cos(self.heading + beta), 
                                    th.sin(self.heading + beta)])
        self.position = self.position + v * dt
        
        self.heading = self.heading + self.speed * th.sin(beta) / (self.LENGTH / 2) * dt
        self.speed = self.speed + self.action['acceleration'] * dt
        
def auto_vehicle_apply_action(position: th.Tensor,
                                speed: th.Tensor,
                                heading: th.Tensor,
                                length: th.Tensor,
                                steering: th.Tensor,
                                acceleration: th.Tensor,
                                dt: float):

    num_vehicle = len(position)

    assert position.ndim == 2, ""
    assert speed.ndim == 1, ""
    assert heading.ndim == 1, ""
    assert steering.ndim == 1, ""
    assert acceleration.ndim == 1, ""

    assert len(speed) == num_vehicle, ""
    assert len(heading) == num_vehicle, ""
    assert len(steering) == num_vehicle, ""
    assert len(acceleration) == num_vehicle, ""

    max_speed = Vehicle.MAX_SPEED
    min_speed = Vehicle.MIN_SPEED

    min_accel = th.clip(acceleration, min=max_speed - speed)
    max_accel = th.clip(acceleration, max=min_speed - speed)

    t_accel = acceleration
    t_accel = th.where(speed > max_speed, min_accel, t_accel)
    t_accel = th.where(speed < min_speed, max_accel, t_accel)

    delta_f = steering
    beta = th.arctan(1 / 2 * th.tan(delta_f))
    v = speed.unsqueeze(-1) * th.stack([th.cos(heading + beta), th.sin(heading + beta)], dim=1)
    
    n_position = position + v * dt
    n_heading = heading + speed * th.sin(beta) / (length / 2) * dt
    n_speed = speed + t_accel * dt
    n_velocity = n_speed * th.stack([th.cos(n_heading), th.sin(n_heading)], dim=1)

    return n_position, n_velocity, n_heading, n_speed