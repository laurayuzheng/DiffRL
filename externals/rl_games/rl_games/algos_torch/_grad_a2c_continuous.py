from math import floor
from typing import List
from rl_games.common import tr_helpers
import time

import torch 
from torch import nn
import numpy as np
import gym

from rl_games.algos_torch import torch_ext

from rl_games.common.a2c_common import swap_and_flatten01, A2CBase
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common._grad_experience import GradExperienceBuffer
from rl_games.algos_torch._grad_running_mean_std import GradRunningMeanStd
from rl_games.common import schedulers
from rl_games.common import common_losses, _grad_common_losses

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

        # we have additional hyperparameter [gi_alpha] that determines gradient size;

        self.init_max_alpha(config)

        # we have additional hyperparameter [gi_beta] that determines number of virtual actions;

        self.init_beta(config)

        # we have additional hyperparameter [gi_step_num] that determines differentiable step size;

        self.gi_step_num = config.get('gi_step_num', 32)

        # disable some options for now...

        assert not self.has_central_value, "Not supported yet"
        assert not 'phasic_policy_gradients' in self.config, "Not supported yet"

    def init_max_alpha(self, config):

        self.gi_alpha = 0.0
        self.gi_max_alpha: float = config.get('gi_max_alpha', 0.1)
        gi_max_alpha_scheduler_type = config.get('gi_max_alpha_schedule', 'identity')

        if gi_max_alpha_scheduler_type == 'identity':
            self.gi_max_alpha_scheduler = schedulers.IdentityScheduler()
        elif gi_max_alpha_scheduler_type == 'linear':
            self.gi_max_alpha_scheduler = schedulers.LinearScheduler(self.gi_max_alpha, min_lr=0, max_steps=self.max_epochs)
        else:
            raise ValueError()

        self.gi_num_init_search = 32

    def init_beta(self, config):

        self.gi_beta = config.get('gi_beta', 4.0)
        gi_beta_scheduler_type = config.get('gi_beta_schedule', 'linear')

        if gi_beta_scheduler_type == 'identity':
            self.gi_beta_scheduler = schedulers.IdentityScheduler()
        elif gi_beta_scheduler_type == 'linear':
            self.gi_beta_scheduler = schedulers.LinearScheduler(self.gi_beta, min_lr=0, max_steps=self.max_epochs)
        else:
            raise ValueError()

    def update_max_alpha(self):

        self.gi_max_alpha , _ = self.gi_max_alpha_scheduler.update(self.gi_max_alpha, None, self.epoch_num, 0, None)
        self.writer.add_scalar('info/gi_max_alpha', self.gi_max_alpha, self.epoch_num)

    def update_beta(self):

        self.gi_beta , _ = self.gi_beta_scheduler.update(self.gi_beta, None, self.epoch_num, 0, None)
        self.writer.add_scalar('info/gi_beta', self.gi_beta, self.epoch_num)

    def init_alpha(self):

        if self.gi_max_alpha == 0:
            
            self.gi_alpha = 0
            self.writer.add_scalar("info/gi_alpha", self.gi_alpha, self.epoch_num)
            
            return

        alphas: List[float] = np.arange(-self.gi_max_alpha, 
                                        self.gi_max_alpha, 
                                        (2.0 * self.gi_max_alpha / self.gi_num_init_search)).tolist()
        alphas.append(self.gi_max_alpha)
        alphas.append(0.0)

        actions = self.dataset.values_dict['actions']
        advantages = self.dataset.values_dict['advantages']
        adv_grads = self.dataset.values_dict['adv_grads']
        model = self.model
        old_mu = self.dataset.values_dict['mu']
        old_sigma = self.dataset.values_dict['sigma']
        curr_mu = self.dataset.values_dict['mu']
        curr_sigma = self.dataset.values_dict['sigma']

        min_var = -1

        for alpha in alphas:

            curr_var = _grad_common_losses.alpha_variance_loss(actions, advantages, adv_grads, model, old_mu, old_sigma, curr_mu, curr_sigma, alpha)

            if min_var < 0 or curr_var < min_var:

                min_var = curr_var
                self.gi_alpha = alpha

        self.writer.add_scalar("info/gi_alpha", self.gi_alpha, self.epoch_num)

        # update data;

        adv_grads_norm = torch.pow(torch.norm(adv_grads, p=2, dim=1), 2.0)
        old_logstd = torch.log(old_sigma)

        actions = actions + (adv_grads * self.gi_alpha)
        advantages = advantages + (adv_grads_norm * self.gi_alpha)
        neglogp = model.neglogp(actions, old_mu, old_sigma, old_logstd)

        self.dataset.values_dict['actions'] = actions
        self.dataset.values_dict['advantages'] = advantages
        self.dataset.values_dict['old_logp_actions'] = neglogp

    def beta_sampling(self):

        with torch.no_grad():

            num_additional_info = floor(pow(self.gi_beta, self.actions_num))

            if num_additional_info == 0:
                return

            model = self.model
        
            old_values: torch.Tensor = self.dataset.values_dict['old_values']
            old_logp_actions: torch.Tensor = self.dataset.values_dict['old_logp_actions']
            actions: torch.Tensor = self.dataset.values_dict['actions']
            returns: torch.Tensor = self.dataset.values_dict['returns']
            obs: torch.Tensor = self.dataset.values_dict['obs']
            advantages: torch.Tensor = self.dataset.values_dict['advantages']
            adv_grads: torch.Tensor = self.dataset.values_dict['adv_grads']
            mu: torch.Tensor = self.dataset.values_dict['mu']
            sigma: torch.Tensor = self.dataset.values_dict['sigma']
            logstd = torch.log(sigma)
            
            # sample new actions using [mu] and [sigma];

            base_old_values = old_values.repeat((num_additional_info, 1))
            base_old_logp_actions = old_logp_actions.repeat((num_additional_info,))
            base_actions = actions.repeat((num_additional_info, 1))
            base_returns = returns.repeat((num_additional_info, 1))
            base_obs = obs.repeat((num_additional_info, 1))
            base_advantages = advantages.repeat((num_additional_info,))
            base_adv_grads = adv_grads.repeat((num_additional_info, 1))
            base_mu = mu.repeat((num_additional_info, 1))
            base_sigma = sigma.repeat((num_additional_info, 1))
            base_logstd = logstd.repeat((num_additional_info, 1))

            # base_distr = torch.distributions.Normal(base_mu, base_sigma)

            # additional_actions = base_distr.sample()
            
            # additional_old_logp_actions = model.neglogp(additional_actions, base_mu, base_sigma, base_logstd)

            # additional_actions_diff = additional_actions - base_actions

            additional_mu = torch.zeros_like(base_mu)
            additional_sigma = torch.ones_like(base_sigma)
            base_actions_size = torch.norm(base_actions, p=2, dim=1, keepdim=True)
            additional_sigma *= base_actions_size

            additional_distr = torch.distributions.Normal(additional_mu, additional_sigma)

            additional_actions_diff = additional_distr.sample()

            additional_actions = base_actions + additional_actions_diff

            additional_old_logp_actions = model.neglogp(additional_actions, base_mu, base_sigma, base_logstd)

            additional_advantages_diff = torch.matmul(additional_actions_diff.unsqueeze(-2), base_adv_grads.unsqueeze(-1))
            additional_advantages = base_advantages + additional_advantages_diff.squeeze()

            # concatenate additional info;

            old_values = torch.cat([old_values, base_old_values])
            old_logp_actions = torch.cat([old_logp_actions, additional_old_logp_actions])
            actions = torch.cat([actions, additional_actions])
            returns = torch.cat([returns, base_returns])
            obs = torch.cat([obs, base_obs])
            advantages = torch.cat([advantages, additional_advantages])
            adv_grads = torch.cat([adv_grads, base_adv_grads])
            mu = torch.cat([mu, base_mu])
            sigma = torch.cat([sigma, base_sigma])

            dataset_dict = {}
            dataset_dict['old_values'] = old_values
            dataset_dict['old_logp_actions'] = old_logp_actions
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = obs
            dataset_dict['rnn_states'] = self.dataset.values_dict['rnn_states']
            dataset_dict['rnn_masks'] = self.dataset.values_dict['rnn_masks']
            dataset_dict['mu'] = mu
            dataset_dict['sigma'] = sigma

            self.dataset.update_values_dict(dataset_dict)

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

        # estimator variances;
        a_est_vars = []

        if self.is_rnn:
            raise NotImplementedError()

        # init alpha;

        # self.init_alpha()

        # init beta;

        self.beta_sampling()

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss, a_est_var = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                a_est_vars.append(a_est_var)

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

        # update max alpha;

        # self.update_max_alpha()

        # update beta;

        self.update_beta()

        # log estimator variance;

        self.writer.add_scalar("info/est_var", np.mean(a_est_vars), self.epoch_num)

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

        # indicator for steps that grad computation starts;
        grad_start = torch.zeros_like(self.experience_buffer.tensor_dict['dones'])

        for n in range(self.horizon_length):

            if n % self.gi_step_num == 0:

                # initialize trajectory to cut off gradients;
                # assume [self.vec_env] is [RLGPUEnv];

                self.obs = self.vec_env.env.initialize_trajectory()
                self.obs = self.obs_to_tensors(self.obs)
                grad_start[n, :] = 1.0

            else:

                # if trajectory has been done in last step and new episode starts, cut off grads;
                grad_start[n, :] = self.dones

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
            actions = torch.tanh(self.experience_buffer.tensor_dict['actions'][n])
            # actions = self.experience_buffer.tensor_dict['actions'][n]
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
        mb_adv_grads = self.advantage_gradients_gae(mb_advs, grad_start)
        self.experience_buffer.tensor_dict['adv_grads'] = mb_adv_grads

        # clear computation graph;

        self.clear_experience_buffer_grads()

        with torch.no_grad():

            batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
            batch_dict['returns'] = swap_and_flatten01(mb_returns.detach())
            batch_dict['played_frames'] = self.batch_size
            batch_dict['step_time'] = step_time


        return batch_dict

    def advantage_gradients_gae(self, mb_advs, grad_start):

        '''
        Compute advantage gradients, of which size equals to (# timestep, # actors, # action size).
        '''

        grad_start_indices = grad_start.unsqueeze(-1).nonzero(as_tuple=True)
        adv_sum = torch.sum(mb_advs[grad_start_indices])

        num_timestep = grad_start.shape[0]
        num_actors = grad_start.shape[1]

        # compute gradients;

        adv_sum.backward()

        adv_grads = self.experience_buffer.tensor_dict['actions'].grad

        # reweight grads;

        with torch.no_grad():

            c = (1.0 / (self.gamma * self.tau))
            cv = torch.ones((num_actors, 1), device=adv_grads.device)

            for nt in range(num_timestep):

                # if new episode has been started, set [cv] to 1; 
                grad_start_indices = grad_start[nt].unsqueeze(-1).nonzero(as_tuple=True)
                cv[grad_start_indices] = 1.0

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

        advantages = returns - values

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        # adjust [max_alpha] according to grad size and set proper grad size;

        with torch.no_grad():

            adv_grads = batch_dict['adv_grads']

            # adv_grads_size = torch.norm(adv_grads, p=2, dim=1)
            # actions_size = torch.clip(torch.norm(actions, p=2, dim=1), min=1e-6)
            # adv_grads_ratio = adv_grads_size / actions_size
            # mean_adv_grads_ratio = torch.mean(adv_grads_ratio).item()

            # self.writer.add_scalar("info/adv_grads_ratio", mean_adv_grads_ratio, self.epoch_num)

            # # make size of [adv_grads] rougly equal to [actions];

            # adv_grads_multiplier = 1.0 / max(mean_adv_grads_ratio, 1e-6)
            # adv_grads *= adv_grads_multiplier

        # perturb [actions] and [advantages] using [adv_grads];

        # adv_grads_norm = torch.pow(torch.norm(adv_grads, p=2, dim=1, keepdim=True), 2.0)
        # actions = actions + adv_grads * curr_alpha
        # advantages = advantages + adv_grads_norm * curr_alpha


        # render original and perturbed actions;

        # if self.vec_env.env.visualize:

        #     env = self.vec_env.env

        #     env.render('human', actions - adv_grads * curr_alpha, actions)
        
        # if True:

        #     self.set_eval()
            
        #     with torch.no_grad():

        #         obs_batch = obses
        #         obs_batch = self._preproc_obs(obs_batch)

        #         curr_batch_dict = {
        #             'is_train': True,
        #             'prev_actions': actions, 
        #             'obs' : obs_batch,
        #         }

        #         curr_res_dict = self.model(curr_batch_dict)

        #         neglogpacs = curr_res_dict['prev_neglogp']
        #         neglogpacs = torch.clip(neglogpacs, max=16)    

        #     self.set_train()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                raise NotImplementedError()
            else:
                raise NotImplementedError()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

        dataset_dict['adv_grads'] = adv_grads

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            raise NotImplementedError()

    def get_full_state_weights(self):
        
        state = super().get_full_state_weights()

        state['gi_max_alpha'] = self.gi_max_alpha

        return state

    def set_full_state_weights(self, weights):
        
        super().set_full_state_weights(weights)

        self.gi_max_alpha = weights['gi_max_alpha']

    
    def calc_gradients(self, input_dict):

        # =================================================

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = common_losses.actor_loss(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            # compute variance of our estimator;

            if True:

                ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
                mean_terms = advantage * ratio

                _a_est_mean = torch.mean(mean_terms, dim=0).item()
                
                var_terms = torch.pow(mean_terms - _a_est_mean, 2.0)
                _a_est_var = torch.mean(var_terms, dim=0).item()

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            b_loss = self.bound_loss(mu)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss, _a_est_var)