import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Subset
import json

def tokenize_prompt_and_output (prompt_strs, output_strs, tokenizer):
    result = {}

    prompt_tokens = tokenizer (prompt_strs)['input_ids'] #We don't need the attn mask o/p
    output_tokens = tokenizer (output_strs)['input_ids'] #We don't need the attn mask o/p
    #The above are list of lists. [ [tokens of 1st str], [], [] ... [tokens of nth str]]

    #Concat the two! 
    num_strs = len(prompt_strs)
    agg_tokens = []
    agg_mask = []
    for strnum in range(num_strs):
        agg_token = prompt_tokens[strnum].copy() #copy() so that we don't update orig prompt_tokens list!
        agg_token.extend (output_tokens[strnum]) #append the generation tokens!
        agg_token = torch.tensor (agg_token) #conv to tensor so that we can call pad_sequence below
        agg_tokens.append(agg_token)

        
        #Put 0 for prompt tokens and 1 for output_tokens
        mask = [0 for _ in range(len(prompt_tokens[strnum]))]
        mask.extend ([1 for _ in range(len(output_tokens[strnum]))])
        agg_mask.append (torch.tensor (mask) )

        assert len (agg_tokens[-1]) == len(agg_mask[-1])

    #padding token of this tokenizer!
    pad_token_id = tokenizer.pad_token_id

    #Convert to pytorch tensor with padding!
    agg_tensor = torch.nn.utils.rnn.pad_sequence (agg_tokens, batch_first=True, padding_value=pad_token_id)
    agg_mask =  torch.nn.utils.rnn.pad_sequence (agg_mask, batch_first=True, padding_value=0) #pad 0 to parts of each-seq after response so that all are of eq len

    input_ids = agg_tensor[:, :-1]
    labels = agg_tensor[:, 1:]
    response_mask = agg_mask[:, 1:]

    result['input_ids'] = input_ids
    result['labels'] = labels
    result['response_mask'] = response_mask

    return result


def compute_entropy (logits): #logits : [batchsize, seqlen, vocabsize]

    probs = torch.softmax (logits, dim = -1) #[batchsize, seqlen, vocabsize] #p(x)

    #below is numerically stable way for logprobs. there is a pytorch way torch.xlogy which computes x*log(y) that handles 0s better!
    logprobs = logits - torch.logsumexp (logits, dim = -1, keepdim=True) #[batchsize, seqlen, V] <--logsumexp uses max subtraction trick!
    entropy = -1 * torch.sum ((probs * logprobs), dim = -1) #[batchsize, seqlen]

    #logprobs2 = torch.log (probs)
    #entropy2 = -1 * torch.sum ((probs*logprobs2), dim= -1)

    return entropy

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

def masked_normalize (tensor, mask, normalize_constant, dim): #tensor and mask should be same shape!
    
    masked_tensor = tensor.masked_fill (~mask.bool(), 0)

    summed_tensor = torch.sum (masked_tensor, dim = dim, keepdim=False) 
    #of shape [...,x,...] all orig except one being sumed on which is removed!
    #when dim is None : scalar!

    output = summed_tensor/normalize_constant
    return output


def sft_microbatch_train_step (policy_logprobs, response_mask, grad_acc_steps, 
                               normalize_constant = 1.0):
    
    #1. Compute loss from the policy_logprobs + response_mask while normalizing
    #2. Scale loss [divide by grad_acc_steps]
    #3. Call loss.backward() to do backprop and store the acc gradient for the model params!

    '''
    Policy_logprobs has logprob of the ground truth label in vocab
    CE = 1/n (- Σ p(y) log (p (ŷ)))
    we have log (p (ŷ) for the GT labels whose p(y) is 1
    so we need to just negate and average the policy_logprobs while respecting response mask!
    '''



    agg_loss =  -1*masked_normalize (policy_logprobs, response_mask, normalize_constant, None) #Torch scalar!

    batch_size = response_mask.shape[0]
    num_elements_considered = response_mask[response_mask>0]

    #avg_loss = agg_loss/num_elements_considered.sum() #can also do len(num_elements_considered) when mask is {0,1} 
    #print (f"Averaging over : {num_elements_considered.sum()}")
    avg_loss = agg_loss/batch_size

    #Applying loss scaling
    avg_loss /= grad_acc_steps

    avg_loss.backward()

    return (avg_loss, None)

def readJSONL (filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data



def log_generations (model, val_set, k, ):

    torch.manual_seed (42)

    indices = torch.randperm(len(val_set))[:k]  # first k indices after random permutation of numbers from 0 to len(val_set)
    subset = Subset(val_set, indices)
    val_dataloader = DataLoader (subset, batch_size = 128, shuffle=False, drop_last=True)


    for v_batchnum, (X_v, Y_v) in enumerate(val_dataloader):
        '''
        1. Call llm.generate() on the batch
        2. Parse the output
        3. Call the grader and note accuracy on this batch for diff rewards!
        4. 
        5. 
        '''
        continue

    return

if __name__ == '__main__':
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
                                                 
    prompt_strs = ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?" ,
                      "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", 
                    ]
    output_strs = ["Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.#### 72",
                    "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10",
                      "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5"
                      ]
    
    result = tokenize_prompt_and_output (prompt_strs, output_strs, tokenizer)

    print (f"input id shape  = {result['input_ids'].shape}")
    print (f"labels shape  = {result['labels'].shape}")
    print (f"response mask shape  = {result['response_mask'].shape}") #parts of the label that are prompt or padding are masked with 0!

    response_logprobs = get_response_log_probs (model, result["input_ids"], result["labels"], True)
    print (f"response log probs shape = {response_logprobs['log_probs'].shape}")
