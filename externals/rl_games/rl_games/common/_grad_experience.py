import torch

from rl_games.common.experience import ExperienceBuffer

class GradExperienceBuffer(ExperienceBuffer):

    def _init_from_env_info(self, env_info):

        super()._init_from_env_info(env_info)

        # also store advantage gradient info;

        self.tensor_dict['adv_grads'] = torch.zeros_like(self.tensor_dict['actions'])