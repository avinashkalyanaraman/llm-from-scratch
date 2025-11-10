import torch
import utils, vllm_helper
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import sys
import grader.drgrpo_grader
from vllm import SamplingParams



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

    SEED = 42

    data_file = 'data/sft.jsonl'
    torch.manual_seed (SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    #Model params!
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype = torch.bfloat16) 

    #SFT-model params!                                           
    #Move model to device
    model.to(device)

    if device.type == 'cuda':
        model = torch.compile (model)

    #Optimizer
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW ( model.parameters(), lr = learning_rate, betas = (0.9,0.999), eps=1e-8, weight_decay = 1e-2)

    #Load the "base model" onto a separate GPU for validation.
    vllm_valdn_model = vllm_helper.init_vllm (model_id, device="cuda:1", seed=SEED)
    #Set sampling params for the validn model's generation!!
    sampling_params = SamplingParams(max_tokens=4096, temperature=1.0, top_p = 1.0, stop=["</answer>"])
    sampling_params.include_stop_str_in_output = True #</answer> will be incl. in generation

    #Epochs!
    num_epochs= 10
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

    #val_tokenized_result = utils.tokenize_prompt_and_output (val_prompt_strs, val_output_strs , tokenizer)
    
    print (f"Shape of training inputs = {train_tokenized_result['input_ids'].shape}")
    #print (f"Shape of val inputs = {val_tokenized_result['input_ids'].shape}")
    print ("----"*20)


    train_data = MyDataset (train_tokenized_result['input_ids'], train_tokenized_result['labels'], train_tokenized_result['response_mask'])
    #val_data = MyDataset (val_tokenized_result['input_ids'], val_tokenized_result['labels'], val_tokenized_result['response_mask'])

    print (f"Train data len = {len(train_data)}")
    #print (f"Val data len = {len(val_data)}")

    train_batch_size = 8
    train_dataloader = DataLoader (train_data, batch_size=train_batch_size, shuffle=True, drop_last=True)
    #val_dataloader = DataLoader (val_data, batch_size=1, shuffle=True, drop_last=True)



    for epoch in range(num_epochs):
        print (f"epoch nunm = {epoch}")
        for batch_num, (X,Y, mask) in enumerate(train_dataloader):
            print (f"Handling batch_num = {batch_num} for epoch {epoch}")
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)


            #Call the model!
            response_logprobs = utils.get_response_log_probs (model, X, Y, False)
            #print (f"Obtained logprobs!")
            avg_loss, metadata = utils.sft_microbatch_train_step (response_logprobs['log_probs'], mask, grad_acc_steps, normalize_constant=1.0)

            #print (f"observed loss = {avg_loss}")


            #Run gradient accumulation!
            if (batch_num + 1) % grad_acc_steps == 0:
                #torch.cuda.empty_cache()
                
                #Clip Gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True) #Faster!
                #print ("1 batch done!")
                print (f"Last observed loss = {avg_loss}")
                print ("--"*20)

            
        #Run validation test
        if (epoch + 1)% 10 == 0:
            #1. Copy current-sft weights to VLLMs GPU (device=cuda:1)
            vllm_helper.load_policy_into_vllm_instance (model, vllm_valdn_model)

            #2 Run generation on it with given validation prompts
            outputs = vllm_valdn_model.generate(val_prompt_strs, sampling_params)

            #3.Grade and see what is our validation result!
            results = utils.evaluate_model (grader.drgrpo_grader.r1_zero_reward_fn, outputs, val_output_strs)

            #4. Look at results to compute valdn_acc
            format_corrects_acc = len(results)*100./len([ele for ele in results if ele['format_reward'] > 0])
            answer_corrects_acc = len(results)*100./len([ele for ele in results if ele['answer_reward'] > 0])

            print(f"VALN :: At epoch : {epoch}, the #format corrects = {format_corrects_acc:0.2f}, "
                    f"#answer_corrects = {answer_corrects_acc:0.2f}")
                

    output_model_path = f"sft_model_bs{train_batch_size}_lr{learning_rate}"
    model.save_pretrained("./sft_model")
