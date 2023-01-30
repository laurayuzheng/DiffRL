from envs.dflex_env import DFlexEnv
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

from envs.traffic.single_pace_car.simulation import PaceCarSim
from envs.traffic.single_pace_car.viewer import PaceCarEnvViewer

class TrafficSinglePaceCarEnv(DFlexEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=64, seed=0, episode_length=1000, no_grad=True, stochastic_init=False, MM_caching_frequency = 1, early_termination = False,
                num_idm_vehicle=9, num_lane=3, speed_limit=20.0, no_steering=False):

        self.num_auto_vehicle = 1
        self.num_idm_vehicle = num_idm_vehicle
        self.num_lane = num_lane
        self.speed_limit = speed_limit
        self.desired_speed_limit = speed_limit * 0.5
        self.no_steering = no_steering

        self.steering_bound = np.deg2rad(20.0)
        if no_steering:
            self.steering_bound = 0.0
        self.acceleration_bound = 8.0

        # pos, vel, idm properties;
        self.num_obs_per_vehicle = 2 + 2

        # steering, accelerations;
        self.num_action_per_vehicle = 2

        num_obs = (num_idm_vehicle + self.num_auto_vehicle) * self.num_obs_per_vehicle
        num_act = self.num_auto_vehicle * self.num_action_per_vehicle

        super(TrafficSinglePaceCarEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)

        self.stochastic_init = stochastic_init
        self.early_termination = early_termination

        self.viewer = None

        self.init_sim()

    def init_sim(self):
        
        self.dt = 0.03
        self.sim = PaceCarSim(self.num_envs, 
                                self.num_auto_vehicle, 
                                self.num_idm_vehicle, 
                                self.num_lane,
                                self.speed_limit,
                                self.desired_speed_limit,
                                self.no_steering,
                                self.device)
        
    def render(self, mode = 'human'):
        # render only first env;
        if self.visualize:
            env = self.sim # self.sim.env_list[0]

            if self.viewer is None:
                config = {'screen_width': 720, 'screen_height': 480, 'offscreen_rendering': False}
                self.viewer = PaceCarEnvViewer(env, config)

            self.enable_auto_render = True

            # add vehicles into the scene;
            observer = env.add_vehicles_to_road_for_view()
            self.viewer.observer_vehicle = observer
            
            env.road = env.hw_road
            env.config = {"simulation_frequency": 30, "real_time_rendering": True}
            env.observation_type = None

            mean_speed = self.sim.vehicle_speed[0, self.num_auto_vehicle:].mean().cpu().item()
            self.viewer.display(mean_speed)

            del env.road
            del env.config
            del env.observation_type

            if mode == 'rgb_array':
            
                image = self.viewer.get_image()
                return image
    
    def step(self, actions: torch.Tensor):
        with df.ScopedTimer("simulate", active=False, detailed=False):
            actions = actions.view((self.num_envs, self.num_actions))
            
            actions = torch.clip(actions, -1., 1.)
            actions[:, 0::2] = actions[:, 0::2] * self.steering_bound
            actions[:, 1::2] = actions[:, 1::2] * self.acceleration_bound
            
            self.actions = actions
            
            self.sim.forward(actions, self.dt)
            self.sim_time += self.dt
            
        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'episode_end': self.termination_buf
                }

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        with df.ScopedTimer("reset", active=False, detailed=False):
            if len(env_ids) > 0:
                self.reset(env_ids)
        
        with df.ScopedTimer("render", active=False, detailed=False):
            self.render()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            self.sim.reset_env(env_ids)
            
            self.progress_buf[env_ids] = 0

            self.calculateObservations()

        return self.obs_buf

    '''
    cut off the gradient from the current state to previous states
    '''
    def clear_grad(self):
        with torch.no_grad():
            self.sim.clear_grad()

    '''
    This function starts collecting a new trajectory from the current states but cut off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and return the observation vectors
    '''
    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def calculateObservations(self):
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)

        position_x = self.sim.vehicle_world_position[:, :, 0] - self.sim.vehicle_world_position[:, [0], 0]
        position_y = self.sim.vehicle_world_position[:, :, 1] - self.sim.vehicle_world_position[:, [0], 1]
        velocity_x = self.sim.vehicle_world_velocity[:, :, 0]
        velocity_y = self.sim.vehicle_world_velocity[:, :, 1]

        self.obs_buf = torch.cat([position_x, position_y, velocity_x, velocity_y], dim=1)

        return

    def calculateReward(self):

        self.rew_buf = self.rew_buf.detach()

        # average disparity to desired speed of idm vehicles;
        abs_idm_vehicle_speed_diff = torch.abs(self.sim.vehicle_speed[:, self.num_auto_vehicle:] - self.desired_speed_limit).mean(dim=1) #[0]
        abs_idm_vehicle_speed_diff = torch.clamp(abs_idm_vehicle_speed_diff / self.desired_speed_limit, max=1.0)
        self.rew_buf = (1.0 - abs_idm_vehicle_speed_diff)

        # reset agents
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # reset collided envs;
        collided = self.sim.check_auto_collision()
        self.reset_buf[collided] = 1.0
        self.rew_buf[collided] = -1.0

        # reset out of lane;
        outoflane = self.sim.check_auto_outoflane()
        self.reset_buf[outoflane] = 1.0
        self.rew_buf[outoflane] = -1.0

        # reset too far envs;
        too_far = self.sim.check_auto_too_far()
        self.reset_buf[too_far] = 1.0
        self.rew_buf[too_far] = -1.0

        # reset auto vehicle in behind;
        behind = self.sim.check_auto_behind()
        self.reset_buf[behind] = 1.0
        self.rew_buf[behind] = -1.0