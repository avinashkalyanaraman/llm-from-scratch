import torch
from torch.utils.data import DataLoader, Dataset
import utils

from transformers import AutoTokenizer, AutoModelForCausalLM

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grader.drgrpo_grader
from vllm import SamplingParams
import vllm_helper

import argparse
import wandb

IS_WANDB = True

class MyDataset(Dataset):
    def __init__(self, x, mask, adv_rewards, agg_rewards, prompt_token_lens, compln_token_lens, logprob_matrix, logprob_response_mask):
        self.x = x[...,:-1]
        self.y = x[..., 1:]
        self.mask = mask[..., :-1]
        self.adv_rewards = adv_rewards
        self.agg_rewards = agg_rewards
        self.prompt_token_lens = prompt_token_lens
        self.compln_token_lens = compln_token_lens
        self.logprob_matrix = logprob_matrix
        self.logprob_response_mask = logprob_response_mask

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx], self.mask[idx], self.adv_rewards[idx], self.agg_rewards[idx], 
                self.prompt_token_lens[idx], self.compln_token_lens[idx], self.logprob_matrix[idx], self.logprob_response_mask[idx])
    
    def __len__(self):
        return self.x.shape[0]


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="GPRO Run")
    parser.add_argument("--lr", default = 1e-5, type=float, help="Learning rate")
    parser.add_argument("--n_grpo_steps", type=int, default=200, help="Number of times to do rollouts (Outerloop)")
    parser.add_argument("--advantage_eps", type=float, default=1e-6, help="epsilon to avoid div by zero in grpo norm")
    
    parser.add_argument("--num_training_samples", type=int, default=1024, help="# training samples to take from the training file [=# of prompts]")
    
    parser.add_argument("--group_size", type=int, default=8, help="groupsize : number of rollouts per prompt")
    
    parser.add_argument("--sampling_temperature", default = 1, type=float, help="Temperature while sampling")
    parser.add_argument("--sampling_min_tokens", type=int, default=4, help="min output token len for vllm rollout")
    parser.add_argument("--sampling_max_tokens", type=int, default=4096, help="max output token len for vllm rollout")

    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1, help="off-policy/near on-policy len")

    parser.add_argument("--train_batch_size", type=int, default=256, help="training batch size (# of prompts in a batch. each prompt is rolled out group_size times)")
    #parser.add_argument("--rollout_batch_size", type=int, default=256, help="rollout batch size (# of rollouts in a batch.")
    parser.add_argument("--gradient_acc_steps", type=int, default=128, help="microbatchsize = train_batch_size/gradient_acc_steps")

    parser.add_argument("--gpu_mem_utilizn", type=float, default=0.85, help="vllm gpu mem utilzn limit")

    parser.add_argument("--loss", type=str, default="no_baseline", help="loss type to use [no_baseline: REINFORCE vanilla, reinforce_with_baseline, grpo_clip]")
    parser.add_argument("--std_norm", action="store_true", help="do stdev normalization")

    parser.add_argument("--tfile", type=str, default="../sft/data/sft_train.jsonl", help="file to be used as training")
    parser.add_argument("--vfile", type=str, default="../sft/data/sft_valdn.jsonl", help="file to be used for validation")


    args = parser.parse_args()
    learning_rate = args.lr
    n_grpo_steps = args.n_grpo_steps
    advantage_eps = args.advantage_eps
    group_size = args.group_size

    sampling_temperature = args.sampling_temperature
    sampling_min_tokens = args.sampling_min_tokens
    sampling_max_tokens = args.sampling_max_tokens

    epochs_per_rollout_batch = args.epochs_per_rollout_batch

    train_batch_size = args.train_batch_size
    #rollout_batch_size = args.rollout_batch_size
    gradient_acc_steps = args.gradient_acc_steps; assert train_batch_size % gradient_acc_steps == 0
    microbatchsize_train = train_batch_size//gradient_acc_steps
    #microbatchsize_rollout = rollout_batch_size//gradient_acc_steps

    gpu_mem_utilizn = args.gpu_mem_utilizn
    loss_type = args.loss; assert loss_type in ['no_baseline', 'reinforce_with_baseline', 'grpo_clip']
    is_std_norm = args.std_norm

    train_data_file = args.tfile
    valdn_data_file = args.vfile

    num_training_samples = args.num_training_samples

    SEED = 42
    torch.manual_seed (SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    
    #Model params!
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(model_id, dtype = torch.bfloat16)#, attn_implementation="flash_attention_2") 

    #Move model to device
    policy.to(device)

    if device.type == 'cuda':
        # Enable TF32 tensor cores for FP32 matmuls/convs
        torch.set_float32_matmul_precision("high")
        #policy = torch.compile (policy)
    

    #Optimizer
    optimizer = torch.optim.AdamW ( policy.parameters(), lr = learning_rate, betas = (0.9,0.95), eps=1e-8, weight_decay = 0.0)
    
    #Load the "base model" onto a separate GPU for generation.
    vllm_gen_model = vllm_helper.init_vllm (model_id, device="cuda:1", seed=SEED, gpu_memory_utilization=gpu_mem_utilizn)
    #Set sampling params for the vllm model's generation!! min_tokens prevents empty string generation!
    sampling_params = SamplingParams(min_tokens = sampling_min_tokens,
        max_tokens=sampling_max_tokens, n = group_size,
        temperature=sampling_temperature, top_p = 1.0, stop=["</answer>"],
        seed = SEED, logprobs = 1)
    sampling_params.include_stop_str_in_output = True #</answer> will be incl. in generation


    #Lets us read the training and validn data files!
    train_data = utils.readJSONL (train_data_file)
    val_data = utils.readJSONL (valdn_data_file)
    print (f"Train data len [original]= {len(train_data)}")
    print (f"Val data len = {len(val_data)}")

    '''
    train_prompt_strs = [ele['prompt'] for ele in train_data]
    train_output_strs = [ele['response'] for ele in train_data]
    val_prompt_strs = [ele['prompt'] for ele in val_data]
    val_output_strs = [ele['response'] for ele in val_data]
    '''

    num_steps = 0

    for on_policy_step in range(n_grpo_steps):

        #Let us sample num_training_samples elements from the training set!
        sampled_train_prompt_strs, sampled_train_output_strs = utils.sampleTrainingData (train_data, num_training_samples)

        #We take the sampled_train_prompt_strs and pass it through vllm!
        vllm_output = vllm_gen_model.generate(sampled_train_prompt_strs, sampling_params)

        all_completions = [c.text for req in vllm_output for c in req.outputs]
        print (f"Total # of generations = {len(all_completions)}")

        #Get the rewards for the generations
        #Repeated ground truth
        agg_sampled_train_output_strs= [ele for ele in sampled_train_output_strs for _ in range(group_size)] #Len = B*G
        assert len(all_completions) == len(agg_sampled_train_output_strs)
        adv_rewards, agg_rewards, _ = utils.compute_group_normalized_rewards (grader.drgrpo_grader.r1_zero_reward_fn, 
                                                          all_completions, agg_sampled_train_output_strs,
                                                          group_size, advantage_eps, is_std_norm) #[B*G,]

        #Get the logprobs that is 0-padded, and the corresponding response mask with mask 0 for pads 
        logprob_matrix, logprob_response_mask = utils.getVLLMLogProbMatrix (vllm_output) #[B*G, max_output_token_len]

        '''
        #Repeated prompts
        agg_sampled_train_prompt_strs = [ele for ele in sampled_train_prompt_strs for _ in range(group_size)] 
        assert len(all_completions) == len(agg_sampled_train_output_strs) #len = B*G
        '''

        #list where each element is two lists -- prompt tokens and compln tokens
        prompt_compln_tokenpairs = [ (ele1.prompt_token_ids,  ele2.token_ids) for ele1 in vllm_output for ele2 in ele1.outputs]        
        off_policy_prompts_complns, off_policy_mask = utils.fusePromptComplnForOffPolicy (prompt_compln_tokenpairs, tokenizer.pad_token_id) #[B*G, S (max_seq_len)]
        prompt_token_lens = torch.tensor ([len(ele) for ele , _ in prompt_compln_tokenpairs]) #shape = [B*G]
        compln_token_lens = torch.tensor ([len(ele) for _ , ele in prompt_compln_tokenpairs]) #shape = [B*G]

        #There are two masks:
        #off_policy_mask : this says amongst the 'S' [max_seq_len tokens] which are generations
        #logprob_response_mask : this says amongst the "S'" [max_gen_len tokens] which are generations


        off_policy_train_data = MyDataset (off_policy_prompts_complns, off_policy_mask, adv_rewards, agg_rewards, 
                                           prompt_token_lens, compln_token_lens, logprob_matrix, logprob_response_mask)
        print (f"Off-policy Train data len = {len(off_policy_train_data)}")
        off_policy_train_dataloader = DataLoader (off_policy_train_data, batch_size=microbatchsize_train, shuffle=True, drop_last=True)

        for off_policy_train_step in range (epochs_per_rollout_batch):
            optimizer.zero_grad(set_to_none=True) #Faster + zero-ing here also handles case when traindata size and accumulated batch size aren't multiples
            #causing the grad-acc if block to not execute and hence not zero-out the gradients.!
            
            #Pass the prompt + completion that we got via VLLM into the policy model, and get the logits!
            for off_policy_batchnum, (X, Y, mask, mub_adv_rewards, mub_agg_rewards, mub_prompt_token_lens, 
                                      mub_compln_token_lens, mub_logprob_matrix, 
                                      mub_logprob_resp_mask) in enumerate (off_policy_train_dataloader):
                
                #Move tensors to device
                X = X.to(device) #[muB, S-1]  where S = max_seq_len (incl prompt + gen)
                Y = Y.to(device) #[muB, S-1]
                mask = mask.to(device) #[muB, S-1]
                mub_adv_rewards = mub_adv_rewards.to(device) #[muB,]
                mub_agg_rewards = mub_agg_rewards.to(device) #[muB,]
                mub_logprob_matrix = mub_logprob_matrix.to(device) #[muB,S'] where S' = max_gen_len
                mub_logprob_resp_mask = mub_logprob_resp_mask.to(device) #[muB,S'] where S' = max_gen_len
                
                #Call the model!
                response_logprobs = utils.get_response_log_probs (policy, X, Y, False) #[muB, S-1]
                
                #Adjust response-logprobs to not inc. the prompt itself, except the last token of the prompt so that we can compare with vllm extracted logprobs!
                adj_response_logprobs = utils.adjustResponseLogProbs (response_logprobs, mub_prompt_token_lens, mub_compln_token_lens) #[muB, max_gen_len]

                #Compute loss for the micro-batch
                mean_per_token_loss, metadata = utils.grpo_microbatch_train_step( adj_response_logprobs, mub_logprob_resp_mask, gradient_acc_steps,
                               loss_type, mub_agg_rewards, mub_adv_rewards, mub_logprob_matrix, advantage_eps)

            
                if (off_policy_batchnum + 1) % gradient_acc_steps == 0:

                    #Clip Gradients
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True) #Faster

                    num_steps += 1

            


    #Copy current policy weights to VLLMs GPU (device=cuda:1)
    vllm_helper.load_policy_into_vllm_instance_orig (policy, vllm_gen_model)
        
