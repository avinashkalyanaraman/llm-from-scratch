import torch
from transformers import AutoTokenizer

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



if __name__ == '__main__':
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
    print (f"response mask shape  = {result['response_mask'].shape}")
