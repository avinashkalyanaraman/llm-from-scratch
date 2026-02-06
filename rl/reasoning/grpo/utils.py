import torch
import json
import random


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

    #But we still compute and return those rewards for logging!
    agg_format_rewards = torch.tensor (agg_format_rewards)
    agg_answer_rewards = torch.tensor (agg_answer_rewards)

    #The actual group-norm!
    mean_rewards = torch.mean(agg_rewards, dim = -1, keepdim=True) #[B,1]
    adv_rewards = agg_rewards - mean_rewards #[B,G]

    if normalize_by_std:
        std_rewards = torch.std (agg_rewards, dim=-1, keepdim=True) #[B,1]
        adv_rewards = adv_rewards/(std_rewards + advantage_eps) #[B,G]

    adv_rewards = adv_rewards.view (len(rollout_responses)) #[B*G]
    agg_rewards = agg_rewards.view (len(rollout_responses)) #[B*G] ; the raw unnormalized reward!
    agg_format_rewards = agg_format_rewards.view (len(rollout_responses)) #[B*G] ; the raw format reward for each rollout
    agg_answer_rewards = agg_answer_rewards.view (len(rollout_responses)) #[B*G] ; the raw answer reward for each rollout

    return adv_rewards, agg_rewards, {'agg_format_rewards' : agg_format_rewards, 'agg_answer_rewards' : agg_answer_rewards}


def compute_naive_policy_gradient_loss (raw_rewards_or_advantages, policy_log_probs):
    #raw_rewards_or_advantages [B,1]
    #policy_log_probs [ B, Seqlen]

    return -raw_rewards_or_advantages * policy_log_probs # [B, seqlen]

def compute_grpo_clip_loss (advantages, policy_log_probs, old_log_probs, cliprange):

    #advantages : [B,1]
    #policy_log_probs : [B, S]
    #old_log_probs : [B, S]


    #policy_probs = torch.exp (policy_log_probs) #[B,S]
    #old_probs = torch.exp (old_log_probs) #[B,S]
    #new_over_old_ratio = policy_probs/old_probs


    #more elegant: 1 exp + prevents underflow!
    new_over_old_ratio = torch.exp (policy_log_probs - old_log_probs)
    clipped_new_over_old_ratio = torch.clamp ( new_over_old_ratio , 1-cliprange, 1+cliprange)

    lhs = new_over_old_ratio * advantages
    rhs = clipped_new_over_old_ratio * advantages

    loss = torch.min (lhs, rhs)

    mask = new_over_old_ratio.ne (clipped_new_over_old_ratio) 
    #[B,S] : each elem is true if clipped, false otherwise!

    return -loss, {'isclip':mask}

def compute_policy_gradient_loss (policy_log_probs, loss_type, raw_rewards = None,
                                  advantages = None, old_log_probs = None, cliprange = 0.0):
    
    if loss_type == 'no_baseline':
        return compute_naive_policy_gradient_loss (raw_rewards, policy_log_probs) , {}
    elif loss_type == 'reinforce_with_baseline':
        return compute_naive_policy_gradient_loss (advantages, policy_log_probs), {}
    elif loss_type == 'grpo_clip':
        return compute_grpo_clip_loss (advantages, policy_log_probs, old_log_probs, cliprange)
    else: #bad loss_type
        assert False , "Incorrect loss type!"

def masked_mean(tensor, mask, dim = None):

    mask_compliant_tensor = tensor * mask 
    sum_mask_compliant_tensor = torch.sum (mask_compliant_tensor, dim = dim)
    num_ones_per_dim_in_mask = torch.sum(mask, dim = dim)
    mean_mask_compliant_tensor = sum_mask_compliant_tensor / num_ones_per_dim_in_mask

    return mean_mask_compliant_tensor

def grpo_microbatch_train_step( policy_log_probs, response_mask, gradient_accumulation_steps,
                               loss_type, raw_rewards, advantages, old_log_probs, cliprange):
        

        per_token_loss, metadata = compute_policy_gradient_loss (policy_log_probs, loss_type, raw_rewards, advantages, 
                                      old_log_probs, cliprange)
        
        #Let us ignore loss computation on the padding tokens!
        per_token_loss = masked_mean (per_token_loss, response_mask)

        mean_per_token_loss = torch.mean (per_token_loss, dim = 0) #Scalar . dim=0 is batch dim
        mean_per_token_loss = mean_per_token_loss / gradient_accumulation_steps #gradient accumulation!

        mean_per_token_loss.backward()
        return mean_per_token_loss, metadata

def readJSONL (filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def getVLLMLogProbMatrix (outputs):

    agg_logprobs = []
    generation_lens = []

    for req in outputs: #for each prompt
        for gen in req.outputs : # for each generation within it
            logprobs = gen.logprobs #the logprobs from vllm for this generation
            logprobs = [list(ele.values())[0].logprob for ele in logprobs] #get the top-token's logprob value!
            agg_logprobs.append (torch.tensor (logprobs))
            generation_lens.append (len(logprobs)) #the length of the output generation!
    
    max_gen_len = max(generation_lens)
    agg_mask_tensor = torch.zeros ( len (generation_lens), max_gen_len)
    for gen_num, g_len in enumerate (generation_lens):
        agg_mask_tensor[gen_num, :g_len] = 1

    
    agg_logprobs_tensor = torch.nn.utils.rnn.pad_sequence (agg_logprobs, 
                                                           batch_first=True, 
                                                           padding_value=0)

    return agg_logprobs_tensor, agg_mask_tensor


def evaluate_model (grader_fn, generations, exp_output_strs):
    results = []

    for generation, solution in zip(generations, exp_output_strs):
        result = grader_fn (generation, solution)
        results.append ( (generation, solution, result) )

    return results


def fusePromptComplnForOffPolicy (prompt_compln_tokenpairs, pad_token_id):

    fused = []
    mask = []
    for prompt, compln in prompt_compln_tokenpairs:
        fused_elem = prompt.copy()
        fused_elem.extend (compln)
        fused.append (torch.tensor (fused_elem))

        _mask = [0 for _ in range (len(prompt) - 1)]
        _mask.extend ( [1 for _ in range (len(compln))])
        _mask.append (0) #We add 0 so that we don't want to look at the next token of the last token for eval!
        mask.append (torch.tensor (_mask))

    fused_tensor = torch.nn.utils.rnn.pad_sequence (fused, batch_first=True, padding_value=pad_token_id)
    masked_tensor = torch.nn.utils.rnn.pad_sequence (mask, batch_first=True, padding_value=0)

    return fused_tensor, masked_tensor

# from SFT code!
def compute_entropy (logits): #logits : [batchsize, seqlen, vocabsize]

    probs = torch.softmax (logits, dim = -1) #[batchsize, seqlen, vocabsize] #p(x)

    #below is numerically stable way for logprobs. there is a pytorch way torch.xlogy which computes x*log(y) that handles 0s better!
    logprobs = logits - torch.logsumexp (logits, dim = -1, keepdim=True) #[batchsize, seqlen, V] <--logsumexp uses max subtraction trick!
    entropy = -1 * torch.sum ((probs * logprobs), dim = -1) #[batchsize, seqlen]

    #logprobs2 = torch.log (probs)
    #entropy2 = -1 * torch.sum ((probs*logprobs2), dim= -1)

    return entropy

# from SFT code!
def get_response_log_probs (model, input_ids, labels, return_token_entropy):
    '''
    Returns the logprob of the GT label
    and optionally the entropy of the token distribution
    '''
    #Call model on input_ids

    logits = model (input_ids).logits #[batchsize, seqlen, vocabsize]
    agg_logprobs = logits - torch.logsumexp (logits, dim = -1, keepdim=True) #[batchsize, seqlen, V] <--logsumexp uses max subtraction trick!
    logprobs = torch.gather (agg_logprobs, dim = -1, index = labels.unsqueeze(-1)) #labels made to [b,s,1] . o/p logprobs is [b,s,1]
    logprobs = logprobs.squeeze (-1) #[b,s]

    if return_token_entropy:
        return { "log_probs" : logprobs, "token_entropy" : compute_entropy (logits)}
    return {"log_probs": logprobs} #if we weren't requested entropy!


def adjustResponseLogProbs (response_logprobs, mub_prompt_token_lens, mub_compln_token_lens):

    '''
    response_logprobs is of shape : [muB_size, S-1] where S = max_seq_len (seq = prompt + compln)
    mub_prompt_token_lens is a 1D tensor of length muB_size which includes the length of prompt tokens for each prompt in microbatch!
    '''

    mub_size = len(mub_prompt_token_lens)

    adj_response_logprobs = [response_logprobs[mub_num, 
                                               mub_prompt_token_lens [mub_num]-1 : 
                                               mub_prompt_token_lens [mub_num]-1 + mub_compln_token_lens[mub_num]] 
                                               for mub_num in range(mub_size)]
    
    # -1 because the last prompt token is the first compln token and we want that since resp_logprobs is shifted by 1!
    # and then we take all the tokens that were generated (in the second half of the indexing)!

    adj_response_logprobs = torch.nn.utils.rnn.pad_sequence (adj_response_logprobs, batch_first=True, padding_value=0)

    return adj_response_logprobs

#With LLM-assist! [Compare speed-up]
def adjust_response_logprobs_fast(response_logprobs, prompt_lens, compln_lens):
    B, T = response_logprobs.shape
    device = response_logprobs.device

    Cmax = int(compln_lens.max().item())
    starts = prompt_lens - 1

    idx = starts.unsqueeze(1) + torch.arange(Cmax, device=device).unsqueeze(0)
    idx = idx.clamp(0, T - 1)

    mask = torch.arange(Cmax, device=device).unsqueeze(0) < compln_lens.unsqueeze(1)

    out = response_logprobs.gather(1, idx)
    out = out * mask.to(out.dtype)

    return out



def sampleTrainingData (train_data, num_training_samples):
    #Let us sample num_training_samples elements from the training set!
    indices = random.sample (range (len(train_data)), num_training_samples)
    sampled_train_data = [train_data[ele] for ele in indices]

    sampled_train_prompt_strs = [ele['prompt'] for ele in sampled_train_data]
    sampled_train_output_strs = [ele['response'] for ele in sampled_train_data]

    return sampled_train_prompt_strs, sampled_train_output_strs