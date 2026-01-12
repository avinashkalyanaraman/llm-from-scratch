import torch
import utils
import random

from transformers import AutoTokenizer, AutoModelForCausalLM

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grader.drgrpo_grader
from vllm import SamplingParams
import vllm_helper

import argparse
import wandb

IS_WANDB = True

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
    parser.add_argument("--rollout_batch_size", type=int, default=256, help="rollout batch size (# of rollouts in a batch.")
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
    rollout_batch_size = args.rollout_batch_size
    gradient_acc_steps = args.gradient_acc_steps; assert train_batch_size % gradient_acc_steps == 0
    microbatchsize_train = train_batch_size//gradient_acc_steps
    microbatchsize_rollout = rollout_batch_size//gradient_acc_steps

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


    train_prompt_strs = [ele['prompt'] for ele in train_data]
    train_output_strs = [ele['response'] for ele in train_data]
    val_prompt_strs = [ele['prompt'] for ele in val_data]
    val_output_strs = [ele['response'] for ele in val_data]


    for on_policy_step in range(n_grpo_steps):

        #Let us sample num_training_samples elements from the training set!
        indices = random.sample (range (len(train_data)), num_training_samples)
        sampled_train_data = [train_data[ele] for ele in indices]

        sampled_train_prompt_strs = [ele['prompt'] for ele in sampled_train_data]
        sampled_train_output_strs = [ele['response'] for ele in sampled_train_data]

        #We take the sampled_train_prompt_strs and pass it through vllm!
        outputs = vllm_gen_model.generate(sampled_train_prompt_strs, sampling_params)

        print (f"# of outputs = {len(outputs)}")
        print (f"outputs first 5 = {outputs[:5]}")



        #TODO
        #Copy policy onto vllm-model for next iteration
        #load_policy_into_vllm_instance ()
        break
