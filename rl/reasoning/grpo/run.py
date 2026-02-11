import torch
from torch.utils.data import DataLoader, Dataset
import utils, wandb_utils

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

def runVLLMGeneration( vllm_model, prompt_strs, sampling_params):

    #We take the sampled_train_prompt_strs and pass it through vllm!
    vllm_output = vllm_model.generate(prompt_strs, sampling_params)
    all_completions = [c.text for req in vllm_output for c in req.outputs]
    return vllm_output, all_completions

def getRewardsAcc (outputs, gt_outputs):

    results = utils.evaluate_model (grader.drgrpo_grader.r1_zero_reward_fn, outputs, gt_outputs)

    format_corrects_acc = len([ele for ele in results if ele[2]['format_reward'] > 0])*100./len(results)
    answer_corrects_acc = len([ele for ele in results if ele[2]['answer_reward'] > 0])*100./len(results)

    return format_corrects_acc, answer_corrects_acc


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="GPRO Run")
    parser.add_argument("--lr", default = 1e-5, type=float, help="Learning rate")
    parser.add_argument("--n_grpo_steps", type=int, default=200, help="Number of times to do rollouts (Outerloop)")
    parser.add_argument("--advantage_eps", type=float, default=1e-6, help="epsilon to avoid div by zero in grpo norm")
    parser.add_argument("--cliprange", type=float, default=0.2, help="clip epsilon to use in grpo clipping to curtail updates")
    
    parser.add_argument("--num_training_samples", type=int, default=1024, help="# training samples to take from the training file [=# of prompts]")
    parser.add_argument("--num_valn_samples", type=int, default=2000, help="# valn samples to take from the valn file [=# of prompts]")
    
    parser.add_argument("--group_size", type=int, default=8, help="groupsize : number of rollouts per prompt")
    
    parser.add_argument("--sampling_temperature", default = 1, type=float, help="Temperature while sampling")
    parser.add_argument("--sampling_min_tokens", type=int, default=4, help="min output token len for vllm rollout")
    parser.add_argument("--sampling_max_tokens", type=int, default=4096, help="max output token len for vllm rollout")

    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1, help="off-policy/near on-policy len")

    parser.add_argument("--train_batch_size", type=int, default=256, help="training batch size (# of prompts in a batch. each prompt is rolled out group_size times)")
    #parser.add_argument("--rollout_batch_size", type=int, default=256, help="rollout batch size (# of rollouts in a batch.")
    parser.add_argument("--gradient_acc_steps", type=int, default=128, help="microbatchsize = train_batch_size/gradient_acc_steps")

    parser.add_argument("--gpu_mem_utilizn", type=float, default=0.85, help="vllm gpu mem utilzn limit")

    parser.add_argument("--loss", type=str, default="grpo_clip", help="loss type to use [no_baseline: REINFORCE vanilla, reinforce_with_baseline, grpo_clip]")
    parser.add_argument("--std_norm", action="store_true", help="do stdev normalization")

    parser.add_argument("--tfile", type=str, default="../sft/data/sft_train.jsonl", help="file to be used as training")
    parser.add_argument("--vfile", type=str, default="../sft/data/sft_valdn.jsonl", help="file to be used for validation")

    parser.add_argument("--valn_granularity", type=int, default=1, help="how often (in terms of outer-loop grpo steps) should valn be done and sent to W&B")



    args = parser.parse_args()
    learning_rate = args.lr
    n_grpo_steps = args.n_grpo_steps
    advantage_eps = args.advantage_eps
    cliprange = args.cliprange
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
    num_valn_samples = args.num_valn_samples
    valn_granularity = args.valn_granularity

    SEED = 42
    torch.manual_seed (SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    
    #Model params!
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(model_id, dtype = torch.bfloat16)#, attn_implementation="flash_attention_2") 

    #Move model to device
    policy.to(device)

    assert device.type == 'cuda', 'Built to run on cuda device type'
    # Enable TF32 tensor cores for FP32 matmuls/convs
    torch.set_float32_matmul_precision("high")


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

    valn_sampling_params = SamplingParams(min_tokens = sampling_min_tokens,
        max_tokens=sampling_max_tokens, n = 1,
        temperature=sampling_temperature, top_p = 1.0, stop=["</answer>"],
        seed = SEED, logprobs = 1)
    valn_sampling_params.include_stop_str_in_output = True #</answer> will be incl. in generation
    

    #Copy current policy weights to VLLMs GPU (device=cuda:1)
    vllm_helper.load_policy_into_vllm_instance_orig (policy, vllm_gen_model)


    #Lets us read the training and validn data files!
    train_data = utils.readJSONL (train_data_file)
    val_data = utils.readJSONL (valdn_data_file)
    print (f"Train data len [original]= {len(train_data)}")
    print (f"Val data len = {len(val_data)}")

    '''
    train_prompt_strs = [ele['prompt'] for ele in train_data]
    train_output_strs = [ele['response'] for ele in train_data]
    '''
    val_prompt_strs = [ele['prompt'] for ele in val_data]
    val_output_strs = [ele['response'] for ele in val_data]

    val_prompt_strs, val_output_strs = utils.sampleData (val_data, num_valn_samples)

    #Setup WandB
    run = None
    if IS_WANDB:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="avinashkaly-self",
            # Set the wandb project where this run will be logged.
            project="llm-from-scratch-rlvr",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": learning_rate,
                "batchsize" : train_batch_size,
                "grad_acc_steps" : gradient_acc_steps,
                "n_grpo_steps": n_grpo_steps,
                "group_size" : group_size,
                "epochs_per_rollout_batch" : epochs_per_rollout_batch,
                "model" : model_id,
                "loss_type" : loss_type
            },
        )


    num_steps = 0


    #Logging before RLVR!
    _, valn_completions = runVLLMGeneration(vllm_gen_model, val_prompt_strs, valn_sampling_params) 
    format_corrects_acc, answer_corrects_acc = getRewardsAcc (valn_completions, val_output_strs)
    wandb_utils.logDictToWANDB (run, {'valn_format_acc' : format_corrects_acc, 'valn_answers_acc': answer_corrects_acc}, num_steps, IS_WANDB)

    for on_policy_step in range(n_grpo_steps):

        print ("---"*20)
        print (f"Handling on-policy step # {on_policy_step}")

        #Let us sample num_training_samples elements from the training set!
        sampled_train_prompt_strs, sampled_train_output_strs = utils.sampleData (train_data, num_training_samples)

        #We take the sampled_train_prompt_strs and pass it through vllm!
        vllm_output, all_completions = runVLLMGeneration (vllm_gen_model, sampled_train_prompt_strs, sampling_params)
        print (f"Total # of generations = {len(all_completions)}")

        #Get the rewards for the generations
        #Repeated ground truth
        agg_sampled_train_output_strs= [ele for ele in sampled_train_output_strs for _ in range(group_size)] #Len = B*G
        assert len(all_completions) == len(agg_sampled_train_output_strs)
        adv_rewards, agg_rewards, rewards_metadata_pre = utils.compute_group_normalized_rewards (grader.drgrpo_grader.r1_zero_reward_fn, 
                                                          all_completions, agg_sampled_train_output_strs,
                                                          group_size, advantage_eps, is_std_norm) #[B*G,]
        

        #Get the logprobs that is 0-padded, and the corresponding response mask with mask 0 for pads 
        logprob_matrix, logprob_response_mask = utils.getVLLMLogProbMatrix (vllm_output) #[B*G, max_output_token_len]

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
            print (f"Handling off_policy step {off_policy_train_step} in on_policy_step {on_policy_step}")
            losses_since_last_commit = [] #Aggregating losses here to log for every off_policy_train_step!
            avg_grpo_clips_since_last_commit = [] #Aggregating avg grpo clip %s to log for every off_policy_train_step!

            optimizer.zero_grad(set_to_none=True) #Faster + zero-ing here also handles case when traindata size and accumulated batch size aren't multiples in the inner loop!
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
                mub_prompt_token_lens = mub_prompt_token_lens.to(device) #[B*G]
                mub_compln_token_lens = mub_compln_token_lens.to(device) #[B*G]


                #Shape adjustments to aid broadcasts
                mub_adv_rewards = mub_adv_rewards.unsqueeze(-1) #[muB,1]
                mub_agg_rewards = mub_agg_rewards.unsqueeze(-1) #[muB,1]

                
                #Call the model!
                response_logprobs = utils.get_response_log_probs (policy, X, Y, False)['log_probs'] #[muB, S-1]
                
                #Adjust response-logprobs to not inc. the prompt itself, except the last token of the prompt so that we can compare with vllm extracted logprobs!
                adj_response_logprobs = utils.adjustResponseLogProbsFast (response_logprobs, mub_prompt_token_lens, mub_compln_token_lens) #[muB, max_gen_len in that mini-batch]
                #adj_response_logprobs_slow = utils.adjustResponseLogProbs (response_logprobs, mub_prompt_token_lens, mub_compln_token_lens) #[muB, max_gen_len in that mini-batch]
                #assert torch.allclose(adj_response_logprobs, adj_response_logprobs_slow, rtol=1e-5, atol=1e-6)

                #Now adj_response_logprobs is of shape [mub, max_gen_len_in_minibatch]
                #while mub_logprob_matrix and mub_logprob_resp_mask are of shape [mub, max_gen_len_across_all_B*G_rollouts]
                #therefore we need to adjust shape of mub_logprob_matrix and mub_logprob_resp_mask
                mub_logprob_matrix, mub_logprob_resp_mask = utils.adjustOldLogProbs (mub_logprob_matrix, mub_logprob_resp_mask, 
                                                                                     adj_response_logprobs.shape [-1])

                #Compute loss for the micro-batch
                mean_per_token_loss, metadata = utils.grpo_microbatch_train_step( adj_response_logprobs, mub_logprob_resp_mask, gradient_acc_steps,
                               loss_type, mub_agg_rewards, mub_adv_rewards, mub_logprob_matrix, cliprange)
                
                if loss_type == 'grpo_clip':
                    isclip_mask = metadata['isclip']
                    avg_clipped = torch.sum(isclip_mask)/torch.numel(isclip_mask)
                    avg_grpo_clips_since_last_commit.append (avg_clipped.item())
                    

                losses_since_last_commit.append(mean_per_token_loss.item())

                if (off_policy_batchnum + 1) % gradient_acc_steps == 0:

                    #Clipping Gradients and reporting it
                    max_norm = 1.0
                    preclip_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)
                    clipped_norm = torch.sqrt(sum(p.grad.norm()**2 for p in policy.parameters() if p.grad is not None))
                    clip_fraction = float (preclip_norm > max_norm)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True) #Faster

                    num_steps += 1

                    #For every optimizer update, we will write the loss to wandb!
                    #Write loss, grad_norm, TODO : clipping fraction and token entropy!
                    #1. reporting the loss and  
                    #2. grad norm
                    #wandb_utils.logToWANDB (run, 'train_loss_per_opt_update', mean_per_token_loss.item(), num_steps, IS_WANDB)
                    wandb_utils.logDictToWANDB (run, {'train_loss_per_opt_update': mean_per_token_loss.item(),
                                                      'preclip_norm': preclip_norm, 'clipped_norm' : clipped_norm, 
                                                      'clip_fraction' : clip_fraction}, 
                                                      num_steps, IS_WANDB)


            
            # We will also write training accuracy and valdn accuracy.
            # Note that training accuracy is on a set of samples that varies every epoch!
            wandb_utils.logToWANDB (run, 'avg_train_loss', sum(losses_since_last_commit)/len(losses_since_last_commit), num_steps, IS_WANDB)
            if loss_type == 'grpo_clip':
                wandb_utils.logToWANDB (run, 'avg_grpo_clips', sum(avg_grpo_clips_since_last_commit)/len(avg_grpo_clips_since_last_commit), num_steps, IS_WANDB)
            losses_since_last_commit = [] #Resetting this list to accumulate losses for next epoch!
            avg_grpo_clips_since_last_commit = [] #Resetting this list that accumulates avg grpo losses for the batch!
            
        #Run and report training & validation accuracy at the end of every epoch after copying weights to vllm model!
        vllm_helper.load_policy_into_vllm_instance_orig (policy, vllm_gen_model)

        #Re-run on training set & also validation set and log accuracy

        if (on_policy_step + 1)% valn_granularity == 0:

            #Rerun on training set and see how the accuracy has changed!
            _, all_completions = runVLLMGeneration (vllm_gen_model, sampled_train_prompt_strs, sampling_params)
            print (f"Total # of generations while rerunning on training set post off-policy update = {len(all_completions)}")

            #Get the rewards for the generations
            #Repeated ground truth
            _, _, rewards_metadata_post = utils.compute_group_normalized_rewards (grader.drgrpo_grader.r1_zero_reward_fn, 
                                                            all_completions, agg_sampled_train_output_strs,
                                                            group_size, advantage_eps, is_std_norm) #[B*G,]        


            
            wandb_log_tr_rewards_dict = {'agg_format_rewards_pre_off_policy' : torch.mean(rewards_metadata_pre['agg_format_rewards']).item()*100.,
                            'agg_answer_rewards_pre_off_policy' : torch.mean(rewards_metadata_pre['agg_answer_rewards']).item()*100.,
                            'agg_format_rewards_post_off_policy' : torch.mean(rewards_metadata_post['agg_format_rewards']).item()*100.,
                            'agg_answer_rewards_post_off_policy' : torch.mean(rewards_metadata_post['agg_answer_rewards']).item()*100.}
            wandb_utils.logDictToWANDB (run, wandb_log_tr_rewards_dict, num_steps, IS_WANDB)

            #Run validation!
            _, valn_completions = runVLLMGeneration(vllm_gen_model, val_prompt_strs, valn_sampling_params) 
            format_corrects_acc, answer_corrects_acc = getRewardsAcc (valn_completions, val_output_strs)
            wandb_utils.logDictToWANDB (run, {'valn_format_acc' : format_corrects_acc, 'valn_answers_acc': answer_corrects_acc}, num_steps, IS_WANDB)
        