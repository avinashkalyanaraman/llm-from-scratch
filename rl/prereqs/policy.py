'''
Policy as a Probability Distribution

This file introduces the idea of a policy in Reinforcement Learning (RL),
with a synthetic example and a simple LM-based example.

Agenda:
a. Logits --> Softmax --> Probability distribution over actions (tokens).
b. Sampling an action from the distribution.
c. Computing log-probabilities of sampled actions.
d. Entropy as a measure of policy uncertainty (exploration vs. exploitation).
e. Effect of temperature scaling (sharp vs. flat distributions).

Functions:
a. sample_action_from_logits(): sample an action and return its log-probability.
b. entropy_from_logits(): compute entropy of a distribution.

This is a standalone code to build intuition for how LLMs
can be treated as RL policies (state = prefix, action = next token).
'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


torch.manual_seed(42)

def sample_action_from_logits(logits, temperature = 1): #logits is [B,S,V]

    logits = logits [:,-1,:] #Take only the last token!
    logits = logits / temperature #Temperature scaling, if needed 

    probs = torch.softmax(logits, dim=-1) #[B,V]
    logprobs = torch.log(probs) #[B,V]
 
    action = torch.multinomial (probs, 1) #[B,1] #Sample based on the prob distbn. Note that multinomial works only in 1d/2d
    
    chosen_logprobs = torch.gather (logprobs, dim=-1, index=action)
    chosen_probs = torch.gather (probs, dim=-1, index=action)
    return action, chosen_probs, chosen_logprobs


def entropy_from_logits (logits, temperature = 1): #logits is [B,S,V]
    logits = logits [:,-1,:]
    logits = logits/temperature

    probs = torch.softmax (logits, dim=-1) #[B,V]
    logprobs = torch.log(probs) #[B,V]

    entropy = -1 * torch.sum ( (probs * logprobs), dim = -1)
    print (f" mean entropy with temp = {temperature} is {entropy.mean(): 0.2f}")

    return entropy


def policyViaLLM (text, compute_entropy = True, temperature_sweep=False, isPrint = False):

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    #Tokenize the input and call the model!
    inputs = tokenizer(text, return_tensors="pt") 

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    #print (f"logits shape = {logits.shape}")

    temperatures = [0.5, 1, 2] if temperature_sweep else [1]

    for temperature in temperatures:
        if temperature == 1:
            action, chosen_probs, chosen_logprobs = sample_action_from_logits (logits, temperature = temperature)
            ids = action.squeeze(-1).tolist()     # [B] list of ints
        else:
            sweep_action, _, _ = sample_action_from_logits (logits, temperature = temperature)
            ids = sweep_action.squeeze(-1).tolist()     # [B] list of ints
        
        if isPrint:
            print("tokens:", [tokenizer.decode([i]) for i in ids], "with temp : ", temperature) 


    #Compute the mean entropy for this q:
    if compute_entropy:
        entropy_from_logits(logits)


    return action, chosen_probs, chosen_logprobs #Returns the case when temperature = 1
    

if __name__ == '__main__':
    B,S,V = 8, 25, 10
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logits = torch.randn (B,S,V, device=device) #[S] 
    sample_action_from_logits (logits)
    entropy_from_logits (logits, temperature=1)

    print ("Seeing effect of temperature on entropy")
    #Check what happens as temperature varies. 
    # As temp increases, logits flatten and entropy increases! [more surprise]
    #As temp decreases, logits peak and entropy decreases! [less surprise]
    for temp in [0.5, 1 , 2]:
        entropy_from_logits (logits, temperature=temp)
        print ("------"*10)

    policyViaLLM ("The clouds are sunny", compute_entropy = True, temperature_sweep=True, isPrint=True)

    