from rl_games.common import tr_helpers
import time

import torch 
import gym

from rl_games.algos_torch import torch_ext

from rl_games.common.a2c_common import swap_and_flatten01, A2CBase
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common._grad_experience import GradExperienceBuffer
from rl_games.algos_torch._grad_running_mean_std import GradRunningMeanStd

def tensor_normalize(t):

    return (t - t.mean()) / (t.std() + 1e-8)

class GradA2CAgent(A2CAgent):
    def __init__(self, base_name, config):
        
        super().__init__(base_name, config)

        # change to proper running mean std;

        if self.normalize_input:
            if isinstance(self.observation_space, gym.spaces.Dict):
                raise NotImplementedError()
            else:
                self.running_mean_std = GradRunningMeanStd(self.obs_shape).to(self.ppo_device)

        if self.normalize_value:
            self.value_mean_std = GradRunningMeanStd((1,)).to(self.ppo_device)

        # we have additional hyperparameter [alpha] that determines gradient size;

        self.alpha: float = 1e-2

        self.reduced_e_clip = self.e_clip

        # disable some options for now...

        assert not self.has_central_value, "Not supported yet"
        assert not 'phasic_policy_gradients' in self.config, "Not supported yet"

    def init_tensors(self):
        
        super().init_tensors()

        # use specialized experience buffer;
        
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }

        self.experience_buffer = GradExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        # add advantage grad;
        self.tensor_list = self.tensor_list + ['adv_grads']

    def train_epoch(self):

        A2CBase.train_epoch(self)

        self.set_eval()
        play_time_start = time.time()
        
        if self.is_rnn:
            raise NotImplementedError()
        else:
            batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            raise NotImplementedError()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        if self.is_rnn:
            raise NotImplementedError()

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)   

                if self.schedule_type == 'legacy':  
                    if self.multi_gpu:
                        kl = self.hvd.average_value(kl, 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,kl.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,av_kls.item())
                self.update_lr(self.last_lr)
            kls.append(av_kls)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,av_kls.item())
            self.update_lr(self.last_lr)

        if self.has_phasic_policy_gradients:
            self.ppg_aux_loss.train_net(self)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul


    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }

        # grad should flow here because there is state value in [res_dict];
        res_dict = self.model(input_dict)

        # only allow grad to flow in res_dict['values'];
        for k in res_dict.keys():
            if isinstance(res_dict[k], torch.Tensor):
                if k != 'values':
                    res_dict[k] = res_dict[k].detach()
                if k == 'actions':
                    res_dict[k].requires_grad = True
        
        assert res_dict['rnn_states'] == None, "Not supported yet"
        assert not self.has_central_value, "Not supported yet"

        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def get_values(self, obs):
        # grad should flow;
        if self.has_central_value:
            raise NotImplementedError()
        else:
            self.model.eval()
            processed_obs = self._preproc_obs(obs['obs'])
            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : processed_obs,
                'rnn_states' : self.rnn_states
            }
            result = self.model(input_dict)
            value = result['values']

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def play_steps(self):
        epinfos = []
        update_list = self.update_list

        step_time = 0.0

        # initialize trajectory to cut off gradients between episodes;
        # assume [self.vec_env] is [RLGPUEnv];
        self.obs = self.vec_env.env.initialize_trajectory()
        self.obs = self.obs_to_tensors(self.obs)

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            # we store tensor objects with gradients right into buffer,
            # as they will be used later in GAE, which also needs gradient;
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            actions = self.experience_buffer.tensor_dict['actions'][n]
            self.obs, rewards, self.dones, infos = self.env_step(actions)
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            # [DefaultRewardsShaper] is differentiable;
            assert isinstance(self.rewards_shaper, tr_helpers.DefaultRewardsShaper), "Not supported yet"
            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                raise NotImplementedError()
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards.detach()
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        # compute advantage gradients;

        self.experience_buffer.tensor_dict['actions'].retain_grad()
        mb_adv_grads = self.advantage_gradients_gae(mb_advs, mb_fdones)
        self.experience_buffer.tensor_dict['adv_grads'] = mb_adv_grads

        # clear computation graph;

        self.clear_experience_buffer_grads()

        with torch.no_grad():

            batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
            batch_dict['returns'] = swap_and_flatten01(mb_returns.detach())
            batch_dict['played_frames'] = self.batch_size
            batch_dict['step_time'] = step_time


        return batch_dict

    def advantage_gradients_gae(self, mb_advs, mb_fdones):

        '''
        Compute advantage gradients, of which size equals to (# timestep, # actors, # action size).
        '''

        new_episode_indices = mb_fdones.unsqueeze(-1).nonzero(as_tuple=True)
        adv_sum = torch.sum(mb_advs[new_episode_indices])

        num_timestep = mb_fdones.shape[0]
        num_actors = mb_fdones.shape[1]

        # compute gradients;

        adv_sum.backward()

        adv_grads = self.experience_buffer.tensor_dict['actions'].grad

        # reweight grads;

        with torch.no_grad():

            c = (1.0 / (self.gamma * self.tau))
            cv = torch.ones((num_actors, 1), device=adv_grads.device)

            for nt in range(num_timestep):

                # if new episode has been started, set [cv] to 1; 
                new_episode_indices = mb_fdones[nt].unsqueeze(-1).nonzero(as_tuple=True)
                cv[new_episode_indices] = 1.0

                adv_grads[nt] = adv_grads[nt] * cv
                cv = cv * c

        return adv_grads

    def clear_experience_buffer_grads(self):

        '''
        Clear computation graph attached to the tensors in the experience buffer.
        '''

        with torch.no_grad():

            for k in self.experience_buffer.tensor_dict.keys():

                if not isinstance(self.experience_buffer.tensor_dict[k], torch.Tensor):

                    continue

                self.experience_buffer.tensor_dict[k] = self.experience_buffer.tensor_dict[k].detach()

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        adv_grads = batch_dict['adv_grads']

        advantages = returns - values

        # perturb [actions] and [advantages] using [adv_grads];

        adv_grads_norm = torch.pow(torch.norm(adv_grads, p=2, dim=1, keepdim=True), 2.0)
        actions = actions + adv_grads * self.alpha
        advantages = advantages + adv_grads_norm * self.alpha

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        
        # determine [reduced_e_clip];

        self.set_eval()

        with torch.no_grad():

            obs_batch = obses
            obs_batch = self._preproc_obs(obs_batch)

            curr_batch_dict = {
                'is_train': True,
                'prev_actions': actions, 
                'obs' : obs_batch,
            }

            curr_res_dict = self.model(curr_batch_dict)
            action_log_probs = -curr_res_dict['prev_neglogp']
            old_action_log_probs = -neglogpacs

            c = torch.exp(old_action_log_probs - action_log_probs)

            c0 = (1.0 - c + self.e_clip) / c
            c1 = (c - 1.0 + self.e_clip) / c
            c = torch.min(torch.stack([c0, c1], dim=1), dim=1)[0]

            self.reduced_e_clip = torch.min(c).item()
            self.reduced_e_clip = max(self.reduced_e_clip, 1e-2)
            self.reduced_e_clip = min(self.reduced_e_clip, self.e_clip)

            self.writer.add_scalar("info/reduced_e_clip", self.reduced_e_clip, self.frame)
            

        self.set_train()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                raise NotImplementedError()
            else:
                # compute jacobian;
                # adv_jac = torch.autograd.functional.jacobian(tensor_normalize, advantages)
                # adv_jac_diag = torch.diagonal(adv_jac, 0)
                # adv_grads = adv_grads * adv_jac_diag.unsqueeze(-1)

                advantages = tensor_normalize(advantages)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            raise NotImplementedError()

    def get_full_state_weights(self):
        
        state = super().get_full_state_weights()

        state['alpha'] = self.alpha

        return state

    def set_full_state_weights(self, weights):
        
        super().set_full_state_weights(weights)

        self.alpha = weights['alpha']

    
    def calc_gradients(self, input_dict):

        # use [reduced_e_clip] instead;

        e_clip = self.e_clip

        self.e_clip = self.reduced_e_clip

        ans = super().calc_gradients(input_dict)

        self.e_clip = e_clip
        
        return ans