from torch import nn
import torch

def alpha_variance_loss(actions, advantages, adv_grads, model, old_mu, old_sigma, curr_mu, curr_sigma, alpha):
    
    '''
    Compute variance of (alpha-policy) estimator.
    '''

    adv_grads_norm = torch.pow(torch.norm(adv_grads, p=2, dim=1), 2.0)
    
    p_actions = actions + (adv_grads * alpha)
    p_advantages = advantages + (adv_grads_norm * alpha)

    # compute probabilities of [p_actions];

    old_logstd = torch.log(old_sigma)
    curr_logstd = torch.log(curr_sigma)

    old_neglogp = model.neglogp(p_actions, old_mu, old_sigma, old_logstd)
    curr_neglogp = model.neglogp(p_actions, curr_mu, curr_sigma, curr_logstd)

    old_neglogp = torch.squeeze(old_neglogp)
    curr_neglogp = torch.squeeze(curr_neglogp)

    ratio = torch.exp(old_neglogp - curr_neglogp)
    mean_terms = p_advantages * ratio

    a_est_mean = torch.mean(mean_terms, dim=0, keepdim=True)
                
    var_terms = torch.pow(mean_terms - a_est_mean, 2.0)
    a_est_var = torch.mean(var_terms, dim=0).squeeze()

    return a_est_var

def alpha_policy_loss(actions, advantages, adv_grads, model, old_mu, old_sigma, curr_mu, curr_sigma, alpha):
    
    '''
    Compute variance of (alpha-policy) estimator.
    '''

    adv_grads_norm = torch.pow(torch.norm(adv_grads, p=2, dim=1), 2.0)
    
    p_actions = actions + (adv_grads * alpha)
    p_advantages = advantages + (adv_grads_norm * alpha)

    # compute probabilities of [p_actions];

    old_logstd = torch.log(old_sigma)
    curr_logstd = torch.log(curr_sigma)

    old_neglogp = model.neglogp(actions, old_mu, old_sigma, old_logstd)
    curr_neglogp = model.neglogp(p_actions, curr_mu, curr_sigma, curr_logstd)

    old_neglogp = torch.squeeze(old_neglogp)
    curr_neglogp = torch.squeeze(curr_neglogp)

    ratio = torch.exp(old_neglogp - curr_neglogp)
    mean_terms = p_advantages * ratio

    a_est_mean = torch.mean(mean_terms, dim=0, keepdim=True)
                
    var_terms = torch.pow(mean_terms - a_est_mean, 2.0)
    a_est_var = torch.mean(var_terms, dim=0).squeeze()

    return a_est_mean, a_est_var


def alpha_policy_correspondence_loss(actions, advantages, adv_grads, model, old_mu, old_sigma, curr_mu, curr_sigma, alpha):
    
    adv_grads_norm = torch.pow(torch.norm(adv_grads, p=2, dim=1), 2.0)
    
    p_actions = actions + (adv_grads * alpha)
    p_advantages = advantages + (adv_grads_norm * alpha)

    # compute probabilities of [p_actions];

    old_logstd = torch.log(old_sigma)
    curr_logstd = torch.log(curr_sigma)

    old_neglogp = model.neglogp(actions, old_mu, old_sigma, old_logstd)     
    curr_neglogp = model.neglogp(p_actions, curr_mu, curr_sigma, curr_logstd)

    old_neglogp = torch.squeeze(old_neglogp)
    curr_neglogp = torch.squeeze(curr_neglogp)

    ratio = old_neglogp - curr_neglogp      # these two have to be equal;
    loss = torch.norm(ratio, p=2)

    return loss

def alpha_actor_loss(old_action_log_probs_batch, action_log_probs, advantage, is_ppo, curr_e_clip, initial_ratio):

    if is_ppo:
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 
                                initial_ratio * (1.0 - curr_e_clip),
                                initial_ratio * (1.0 + curr_e_clip))
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = (action_log_probs * advantage)
    
    return a_loss