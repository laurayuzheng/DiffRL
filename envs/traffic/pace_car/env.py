from envs.dflex_env import DFlexEnv
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dflex as df

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

from envs.traffic._simulation import TrafficSim
from envs.traffic.pace_car.network import PaceCarRoadNetwork
from highway_env.envs.common.graphics import EnvViewer

class TrafficPaceCarEnv(DFlexEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=64, seed=0, episode_length=1000, no_grad=True, stochastic_init=False, MM_caching_frequency = 1, early_termination = False,
                num_auto_vehicle=1, num_idm_vehicle=1, num_lane=1, speed_limit=20.0, desired_speed_limit=10.0, no_steering=True):

        self.num_auto_vehicle = num_auto_vehicle
        self.num_idm_vehicle = num_idm_vehicle
        self.num_lane = num_lane
        self.speed_limit = speed_limit
        self.desired_speed_limit = desired_speed_limit
        self.no_steering = no_steering

        self.steering_bound = np.deg2rad(90.0)
        if no_steering:
            self.steering_bound = 0.0
        self.acceleration_bound = 40.0

        # pos, vel, idm properties;
        self.num_obs_per_vehicle = 2 + 2 + 6

        # steering, accelerations;
        self.num_action_per_vehicle = 2

        num_obs = (num_idm_vehicle + num_auto_vehicle) * self.num_obs_per_vehicle
        num_act = num_auto_vehicle * self.num_action_per_vehicle

        super(TrafficPaceCarEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)

        self.stochastic_init = stochastic_init
        self.early_termination = early_termination

        self.viewer = None

        self.init_sim()

    def init_sim(self):
        
        self.dt = 0.03
        env_list = []
        for _ in range(self.num_envs):
            env_list.append(PaceCarRoadNetwork(self.speed_limit, 
                                                self.desired_speed_limit,
                                                self.num_auto_vehicle,
                                                self.num_idm_vehicle,
                                                self.num_lane,
                                                self.device))
        self.sim = TrafficSim(env_list)

    def render(self, mode = 'human'):
        # render only first env;
        if self.visualize:
            env = self.sim.env_list[0]

            if self.viewer is None:
                config = {'screen_width': 720, 'screen_height': 480, 'offscreen_rendering': False}
                self.viewer = EnvViewer(env, config)

            self.enable_auto_render = True

            # add vehicles into the scene;
            observer = env.add_vehicles_to_road_for_view()
            self.viewer.observer_vehicle = observer
            
            env.road = env.hw_road
            env.config = {"simulation_frequency": 30, "real_time_rendering": True}
            env.observation_type = None

            self.viewer.display()

            del env.road
            del env.config
            del env.observation_type

            # if not self.viewer.offscreen:
            
            #     self.viewer.handle_events()
            
            if mode == 'rgb_array':
            
                image = self.viewer.get_image()
                return image
    
    def step(self, actions):
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
            for id in env_ids:
                self.sim.env_list[id].reset()
            
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

        for env_id, env in enumerate(self.sim.env_list):

            # idm vehicles;
            for lane in env.lane.values():
                for i in range(lane.num_vehicle()):
                    vid = lane.vehicle_id[i].cpu().item()
                    
                    self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 0] = env.idm_vehicle_position[vid][0]
                    self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 1] = env.idm_vehicle_position[vid][1]
                    self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 2] = env.idm_vehicle_velocity[vid][0]
                    self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 3] = env.idm_vehicle_velocity[vid][1]
                    self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 4] = lane.accel_max[i]
                    self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 5] = lane.accel_pref[i]
                    self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 6] = lane.target_vel[i]
                    self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 7] = lane.min_space[i]
                    self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 8] = lane.time_pref[i]
                    self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 9] = lane.vehicle_length[i]
            
            # auto vehicles;
            for i in range(env.num_auto_vehicle):
                vid = i + env.num_idm_vehicle
                self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 0] = env.auto_vehicle_position[i][0]
                self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 1] = env.auto_vehicle_position[i][1]
                self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 2] = env.auto_vehicle_velocity[i][0]
                self.obs_buf[env_id, vid * self.num_obs_per_vehicle + 3] = env.auto_vehicle_velocity[i][1]

        return

    def calculateReward(self):

        self.rew_buf = self.rew_buf.detach()

        for i, env in enumerate(self.sim.env_list):
            rew = 0
            for lane in env.lane.values():
                curr_vel = lane.curr_vel
                rew = rew + torch.abs(curr_vel - self.desired_speed_limit).sum()
            rew = rew / self.num_idm_vehicle
            self.rew_buf[i] = -rew * 0.01

        # punish out of lane;
        for i, env in enumerate(self.sim.env_list):
            auto_min_offset = env.auto_vehicle_lane_distance.min(dim=1)[0]
            auto_min_offset = torch.clip(auto_min_offset, min=0.)
            punish = auto_min_offset.sum() / self.num_auto_vehicle
            self.rew_buf[i] = self.rew_buf[i] - punish * 0.001
        
        # reset agents
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # reset collided envs;
        for i, env in enumerate(self.sim.env_list):
            collided = env.check_auto_collision()
            if collided:
                self.reset_buf[i] = 1.0
                self.rew_buf[i] = self.rew_buf[i] - 1000.0