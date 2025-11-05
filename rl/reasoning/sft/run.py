import torch
import utils
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import sys


'''
1. Read the sft dataset: sft.jsonl
2. Split the orig. data to  train + val. [80-20] and tokenize them to input + labels
3. Call the model for "E" epochs!
4. Every "k" steps run validation while incl. grader!
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

    data_file = 'data/sft.jsonl'
    torch.manual_seed (42)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    #TODO : RM later
    #device = torch.device('cpu')

    #Model params!
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.float16, 
                                                 attn_implementation = "flash_attention_2")
    if device.type == 'cuda':
        torch.compile (model)
    #Move model to device
    model.to(device)

    #Optimizer
    optimizer = torch.optim.AdamW ( model.parameters(), lr = 1e-3, betas = (0.9,0.999), eps=1e-8, weight_decay = 1e-2)

    #Epochs!
    num_epochs= 10
    grad_acc_steps = 4

    #Read the dataset!
    dataset = utils.readJSONL (data_file)
    print (f"Total Dataset size = {len(dataset)}")


    #Split the data!
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_data = train_data[0:100]
    val_data = val_data[0:8]

    print (f"Train data len = {len(train_data)}")
    print (f"Val data len = {len(val_data)}")
    
    train_prompt_strs = [ele['prompt'] for ele in train_data]
    train_output_strs = [ele['response'] for ele in train_data]
    val_prompt_strs = [ele['prompt'] for ele in val_data]
    val_output_strs = [ele['response'] for ele in val_data]

    train_tokenized_result = utils.tokenize_prompt_and_output (train_prompt_strs, train_output_strs , tokenizer)
    val_tokenized_result = utils.tokenize_prompt_and_output (val_prompt_strs, val_output_strs , tokenizer)
    
    print (f"Shape of training inputs = {train_tokenized_result['input_ids'].shape}")
    print (f"Shape of val inputs = {val_tokenized_result['input_ids'].shape}")
    print ("----"*20)


    train_data = MyDataset (train_tokenized_result['input_ids'], train_tokenized_result['labels'], train_tokenized_result['response_mask'])
    val_data = MyDataset (val_tokenized_result['input_ids'], val_tokenized_result['labels'], val_tokenized_result['response_mask'])

    print (f"Train data len = {len(train_data)}")
    print (f"Val data len = {len(val_data)}")

    train_dataloader = DataLoader (train_data, batch_size=2, shuffle=True, drop_last=True)
    val_dataloader = DataLoader (val_data, batch_size=2, shuffle=True, drop_last=True)


    for epoch in range(num_epochs):
        for itn_num, (X,Y, mask) in enumerate(train_dataloader): #TODO: EDIT
            print (f"Handling itn-num = {itn_num}")
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)


            #Call the model!
            response_logprobs = utils.get_response_log_probs (model, X, Y, True)
            print (f"Obtained logprobs!")
            avg_loss, metadata = utils.sft_microbatch_train_step (response_logprobs['log_probs'], mask, grad_acc_steps, normalize_constant=1.0)

            #Run gradient accumulation!
            if (itn_num + 1) % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True) #Faster!

            
            #Run validation test
            #TODO!

            sys.exit(0)


            



    #Run-training/Call the model!
    response_logprobs = utils.get_response_log_probs (model, val_tokenized_result["input_ids"].to(device), val_tokenized_result["labels"].to(device), True)
    print (f"response log probs shape = {response_logprobs['log_probs'].shape}")



    #Repeat!