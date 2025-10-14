import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


class LLMPolicy (torch.nn.Module):
    def __init__ (self, model, tokenizer, max_seqlen):
        super().__init__()

        self.model = model
        self.max_seqlen = max_seqlen
        self.tokenizer = tokenizer



    def forward (self, prompt):


        #1. Tokenize the prompt
        #2. Pass it through the model for num_flips steps or <EOS> 
        #3. Store logprobs of each chosen action. Store each action taken too
        #4. Return actions & logprobs [actions aren't really used] + total response to search for correctness!


        actions = []
        logprobs = []

    

        for _ in range (self.max_seqlen): #TODO: Handle <eos>

            #Tokenize the prefix thus far
            inputs = self.tokenizer(prompt, return_tensors="pt")  
            #inputs has to fields: input_ids (tensor of tokens), attention_mask (tensor mask of 0/1) . both are shape : [1, #tokens]
 
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            logits = logits[0,-1,:] #treating only one batch, and the predxn of the last token of the current prefix! #[V]
  
            probs = torch.softmax(logits, dim=-1)    #[V]       
            action = torch.multinomial (probs, num_samples=1) #idx [1] : multinomial returns a tensor
            logprob = torch.log (probs[action]) #[1]  avoids item()


            actions.append (action)
            logprobs.append (logprob)

            #action is an int. Convert it to appropriate english token
            token_id = action.squeeze(-1)   
            prompt = prompt + self.tokenizer.decode (token_id)


        actions = torch.cat (actions)
        logprobs = torch.cat (logprobs)

        
        return actions, logprobs, prompt
    
    @classmethod #testing 2 + 2 = 4
    def computeReward (cls, response): #response incl input question too!
        match = re.search(r"2\s*\+\s*2\s*=\s*(\S+)", response)
        if match and match.group(1).strip() == '4': #we are looking for '4'
            return 1
        return 0
    


if __name__ == "__main__":
 
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    max_seqlen = 6

    torch.manual_seed (42)

    policy = LLMPolicy (model, tokenizer, max_seqlen)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)


    epochs = 100
    prompt = "2 + 2 ="


    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")  
        #inputs has to fields: input_ids (tensor of tokens), attention_mask (tensor mask of 0/1) . both are shape : [1, #tokens]
             
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits[0,-1,:] #treating only one batch, and the predxn of the last token of the current prefix! #[V]

        #Greedy sampling
        action = torch.argmax (logits, dim=-1) #Scalar!


        #action is an int. Convert it to appropriate english token
        token_id = action.item() 
        print (f"output pre RL = : {tokenizer.decode ([token_id])}")

    for epoch_num in range(epochs):
        print (f"epoch num = {epoch_num}")


        optimizer.zero_grad()
        actions, logprobs, response = policy(prompt)
  
        reward = LLMPolicy.computeReward (response)
        print (f"reward = {reward} for response = {response}")

        loss = -torch.sum(logprobs) * reward
        print (f"loss =. {loss.item()}")
        print ("--------------------"*3)

        loss.backward()
        optimizer.step()


    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")  
        #inputs has to fields: input_ids (tensor of tokens), attention_mask (tensor mask of 0/1) . both are shape : [1, #tokens]
            
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits[0,-1,:] #treating only one batch, and the predxn of the last token of the current prefix! #[V]

        #Greedy sampling
        action = torch.argmax (logits, dim=-1) #Scalar!


        #action is an int. Convert it to appropriate english token
        token_id = action.item() 
        print (f"output after RL = : {tokenizer.decode ([token_id])}")