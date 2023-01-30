from envs.traffic.single_pace_car.env import TrafficSinglePaceCarEnv

class TrafficSinglePaceCarEnv_C(TrafficSinglePaceCarEnv):

    def __init__(self, render=False, device='cuda:0', num_envs=64, seed=0, episode_length=1000, no_grad=True, stochastic_init=False, MM_caching_frequency = 1, early_termination = False,
                speed_limit=20.0, no_steering=False):

        num_lane = 3
        num_idm_vehicle = 9

        super().__init__(render, device, num_envs, seed, episode_length, no_grad, stochastic_init, MM_caching_frequency,
                            early_termination, num_idm_vehicle, num_lane, speed_limit, no_steering)