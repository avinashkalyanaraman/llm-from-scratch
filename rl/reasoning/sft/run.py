import torch
import utils, vllm_helper
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grader.drgrpo_grader
from vllm import SamplingParams
import argparse
import wandb

IS_WANDB = True
VALN_GRANULARITY = 1


'''
1. Read the sft dataset: sft.jsonl
2. Split the orig. data to  train + val. [80-20] and tokenize training to input + labels
3. Call the model for "E" epochs!
4. Every "k" steps run validation while incl. grader! The valn' loop leverages a second GPU!
5. Remember to clip while training!
'''

class MyDataset(Dataset):
    def __init__(self, x, y, mask):
        self.x = x
        self.y = y
        self.mask = mask
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx], self.mask[idx])
    
    def __len__(self):
        return self.x.shape[0]


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Transformer Run")
    parser.add_argument("--lr", default = 1e-4, type=float, help="Learning Rate")
    parser.add_argument("--batchsize", type=int, default=8, help="batchsize")

    args = parser.parse_args()
    learning_rate = args.lr
    train_batch_size = args.batchsize


    SEED = 42

    data_file = 'data/sft.jsonl'
    torch.manual_seed (SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    #Model params!
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype = torch.bfloat16)#, attn_implementation="flash_attention_2") 

    #SFT-model params!                                           
    #Move model to device
    model.to(device)

    if device.type == 'cuda':
        # Enable TF32 tensor cores for FP32 matmuls/convs
        torch.set_float32_matmul_precision("high")
        
        model = torch.compile (model)

    #Optimizer
    optimizer = torch.optim.AdamW ( model.parameters(), lr = learning_rate, betas = (0.9,0.999), eps=1e-8, weight_decay = 1e-2)

    #Load the "base model" onto a separate GPU for validation.
    vllm_valdn_model = vllm_helper.init_vllm (model_id, device="cuda:1", seed=SEED)
    #Set sampling params for the validn model's generation!!
    sampling_params = SamplingParams(max_tokens=4096, temperature=1.0, top_p = 1.0, stop=["</answer>"])
    sampling_params.include_stop_str_in_output = True #</answer> will be incl. in generation

    #Epochs!
    num_epochs= 100
    grad_acc_steps = 8

    #Read the dataset!
    dataset = utils.readJSONL (data_file)
    print (f"Total Dataset size = {len(dataset)}")

    #Split the data!
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)    
    #train_data = train_data[0:1000]
    #val_data = val_data[0:8]

    print (f"Train data len = {len(train_data)}")
    print (f"Val data len = {len(val_data)}")
    
    train_prompt_strs = [ele['prompt'] for ele in train_data]
    train_output_strs = [ele['response'] for ele in train_data]
    val_prompt_strs = [ele['prompt'] for ele in val_data]
    val_output_strs = [ele['response'] for ele in val_data]

    train_tokenized_result = utils.tokenize_prompt_and_output (train_prompt_strs, train_output_strs , tokenizer) 
    #We don't need to concat prompt + output and tokenize validation set
    #We only need to pass (valn_)prompts to the VLLM llm.generate() 
    #where it will use the base model's tokenizer
    #and it will tokenize and generate output for us to grade!
    
    print (f"Shape of training inputs = {train_tokenized_result['input_ids'].shape}")
    print ("----"*20)


    train_data = MyDataset (train_tokenized_result['input_ids'], train_tokenized_result['labels'], train_tokenized_result['response_mask'])
    print (f"Train data len = {len(train_data)}")
    train_dataloader = DataLoader (train_data, batch_size=train_batch_size, shuffle=True, drop_last=True)


    #Setup WandB
    if IS_WANDB:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="avinashkaly-self",
            # Set the wandb project where this run will be logged.
            project="llm-from-scratch-sft",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": learning_rate,
                "batchsize" : train_batch_size,
                "grad_acc_steps" : grad_acc_steps,
                "epochs": num_epochs,
                "model" : model_id
            },
        )

    num_steps = 0
    losses_since_last_commit = []

    for epoch in range(num_epochs):
        print (f"epoch nunm = {epoch}")
        optimizer.zero_grad(set_to_none=True) #Faster + zero-ing here also handles case when traindata size and accumulated batch size aren't multiples causing the grad-acc if block to not execute and hence not zero-out the gradients.!

        for batch_num, (X,Y, mask) in enumerate(train_dataloader):
            print (f"Handling batch_num = {batch_num} for epoch {epoch}")
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)

            #Call the model!
            response_logprobs = utils.get_response_log_probs (model, X, Y, False)
            avg_loss, metadata = utils.sft_microbatch_train_step (response_logprobs['log_probs'], mask, grad_acc_steps, normalize_constant=1.0)
            #print (f"observed loss = {avg_loss}")
            losses_since_last_commit.append(avg_loss.item())


            #Run gradient accumulation!
            if (batch_num + 1) % grad_acc_steps == 0:
                #torch.cuda.empty_cache()                
                #Clip Gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True) #Faster!
                #print ("1 batch done!")
                print (f"Last observed loss for {grad_acc_steps} grad-accumulated {train_batch_size}-batch = {avg_loss}")
                print ("--"*20)
                num_steps += 1

        #Write to WANDB every epoch!
        if IS_WANDB:        
            run.log ( {"avg_train_loss" : sum(losses_since_last_commit)/len(losses_since_last_commit)}, step = num_steps)
        losses_since_last_commit = []

            
        #Run validation test
        if (epoch + 1)% VALN_GRANULARITY == 0:
            #1. Copy current-sft weights to VLLMs GPU (device=cuda:1)
            vllm_helper.load_policy_into_vllm_instance2 (model, vllm_valdn_model)

            #2 Run generation on it with given validation prompts
            outputs = vllm_valdn_model.generate(val_prompt_strs, sampling_params)

            #3.Grade and see what is our validation result!
            results = utils.evaluate_model (grader.drgrpo_grader.r1_zero_reward_fn, outputs, val_output_strs)

            #4. Look at results to compute valdn_acc
            format_corrects_acc = len([ele for ele in results if ele[2]['format_reward'] > 0])*100./len(results)
            answer_corrects_acc = len([ele for ele in results if ele[2]['answer_reward'] > 0])*100./len(results)

            print(f"VALN :: At epoch : {epoch}, the #format corrects = {format_corrects_acc:0.2f}, "
                    f"#answer_corrects = {answer_corrects_acc:0.2f}")
            
            if IS_WANDB:
                run.log( {"format_corrects_acc" : format_corrects_acc}, step = num_steps)
                run.log( {"answer_corrects_acc" : answer_corrects_acc}, step = num_steps) 
            
            if answer_corrects_acc > 15:
                break

            output_model_path = f"sft_model_bs{train_batch_size*grad_acc_steps}_lr{learning_rate}"
            model.save_pretrained(output_model_path)

    #output_model_path = f"sft_model_bs{train_batch_size*grad_acc_steps}_lr{learning_rate}"
    #model.save_pretrained(output_model_path)