import torch

def compute_group_normalized_rewards (reward_fn, rollout_responses, repeated_ground_truths,
                                      group_size, advantage_eps, normalize_by_std):
    
    agg_format_rewards = []
    agg_answer_rewards = []
    agg_rewards = []

    num_batches = int(len(rollout_responses)/group_size) #each batch = prompt

    for rollout, gt in zip (rollout_responses, repeated_ground_truths):
        reward = reward_fn (rollout, gt)

        agg_format_rewards.append (reward ['format_reward'])
        agg_answer_rewards.append (reward ['answer_reward'])
        agg_rewards.append (reward ['reward'])


    #We only care about the rewards here. We don't really worry about format and answer individually
    agg_rewards = torch.tensor (agg_rewards)
    agg_rewards = agg_rewards.view (num_batches, group_size) #[B, G]

    #The actual group-norm!
    mean_rewards = torch.mean(agg_rewards, dim = -1, keepdim=True) #[B,1]
    adv_rewards = agg_rewards - mean_rewards #[B,G]

    if normalize_by_std:
        std_rewards = torch.std (agg_rewards, dim=-1, keepdim=True) #[B,1]
        adv_rewards = adv_rewards/(std_rewards + advantage_eps) #[B,G]

    adv_rewards = adv_rewards.view (len(rollout_responses)) #[B*G]
    agg_rewards = agg_rewards.view (len(rollout_responses)) #[B*G] ; the raw unnormalized reward!

    return adv_rewards, agg_rewards, {}


def compute_naive_policy_gradient_loss (raw_rewards_or_advantages, policy_log_probs):
    #raw_rewards_or_advantages [B,1]
    #policy_log_probs [ B, Seqlen]

    return -raw_rewards_or_advantages * policy_log_probs # [B, seqlen]

def compute_grpo_clip_loss (advantages, policy_log_probs, old_log_probs, cliprange):

    #advantages : [B,1]
    #policy_log_probs : [B, S]
    #old_log_probs : [B, S]


    policy_probs = torch.exp (policy_log_probs) #[B,S]
    old_probs = torch.exp (old_log_probs) #[B,S]
    new_over_old_ratio = policy_probs/old_probs


    #more elegant: 1 exp + prevents underflow!
    new_over_old_ratio = torch.exp (policy_log_probs - old_log_probs)
    clipped_new_over_old_ratio = torch.clamp ( new_over_old_ratio , 1-cliprange, 1+cliprange)

    lhs = new_over_old_ratio * advantages
    rhs = clipped_new_over_old_ratio * advantages

    loss = torch.min (lhs, rhs)

    mask = new_over_old_ratio.ne (clipped_new_over_old_ratio) 
    #[B,S] : each elem is true if clipped, false otherwise!

    return -loss, {'isclip':mask}