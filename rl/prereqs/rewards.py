
import torch
import re

'''
This code deals with trajectory to compute a reward.
In LLMs, rewards are typically measured at the end -- was the response ok/not okay. Episodic!
So, this code extends policy to compute a "trajectory", and computes the reward at the end.
'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import policy

torch.manual_seed(42)


if __name__ == '__main__':
    '''
    Example 1:
    policy : [p, 1-p] : probability of H, T
    Episode : 3 coin tosses
    Reward : +1 if episode ends in HH, 0 otherwise!
    '''

    B,S,V = 5000000,1,2
    p = 0.5 #p(head) 
    episode_len = 3
    corrects = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logits = torch.empty (B,S,V, device = device, dtype=torch.float32)
    logits[:,:,0] = p
    logits[:,:,1] = 1-p

    actions = []
    for _ in range(episode_len):
        action,_,_ = policy.sample_action_from_logits (logits) #action is [B,1]
        actions.append (action)
    
    actions = torch.cat (actions, dim=-1) #actions is [B, episidelen]
    #print (f"action = {actions.shape}")

    '''
    #Iterative way
    #Check if last two were head [class 0]
    num_rows = actions.shape[0]
    for action in actions: #action is [1,episodelen]
        if ( (action[-1] == action [-2]) and (action[-1] == 0) ): #last two are equal, and they are heads!
            corrects += 1
    '''

    #Pytorch way:
    mask = (actions[:, -2] == actions[:, -1]) & (actions[:, -1] == 0) #Those rows where last two actions[/columns] are the same and = 0.
    mask = mask.to(torch.int8)
    corrects = torch.sum(mask)

    #Note that the true probability is 0.25 when p=0.5. So if you keep increasing B, you will approach 0.25. 
    print (f"Total number of trials = {B}")
    print (f"Total % of good episodes ending in HH =  {corrects*100./B : 0.2f}")

    
    #Let's do this for LLMs
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    llm_episode_len = 5
    llm_num_trials = 100
    llm_corrects = 0


    for _ in range(llm_num_trials):
        prefix = "2 + 2 ="
        for _ in range(llm_episode_len):

            action, _, _ = policy.policyViaLLM (prefix, compute_entropy = False, temperature_sweep=False, isPrint=False)
            id = action.squeeze(-1)     
            prefix = prefix + tokenizer.decode (id)
        
        print (f"Output = {prefix}")
        match = re.search(r"2\s*\+\s*2\s*=\s*(\S+)", prefix)
        if match and match.group(1).strip() == '4': #we are looking for '4'
            print (f"match was : {match.group(1)}")
            llm_corrects += 1
    

    print (f"Total number of llm trials = {llm_num_trials}")
    print (f"Total % of 'good' episodes for LLM=  {llm_corrects*100./llm_num_trials : 0.2f}")





