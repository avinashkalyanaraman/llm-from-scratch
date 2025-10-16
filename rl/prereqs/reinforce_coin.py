import torch

class CoinPolicy (torch.nn.Module):
    def __init__ (self, theta, num_flips):
        super().__init__()

        self.theta = torch.nn.Parameter (torch.tensor ([theta], dtype=torch.float32)) #[1]
        self.num_flips = num_flips


    def forward (self):
 
        p = torch.sigmoid (self.theta) #[1]
        #probs = torch.tensor ( [p, 1-p], dtype=torch.float32, requires_grad=True) #[2] <-- this breaks autograd graph because torch.tensor copies p,1-p and creates a brand new tensor
        probs = torch.cat([p, 1 - p]) #[2]

        print (f"probs = {probs}")

        #Slower iterative version!
        '''
        actions = []
        logprobs = []

        for _ in range (self.num_flips):
            action = torch.multinomial (probs, num_samples=1) #idx [1] : multinomial returns a tensor
            #logprob = torch.log (  probs[action.item()] ).unsqueeze(0) #[1] : the unsqueeze converts scalar to tensor! 
            logprob = torch.log (probs[action]) #[1]  avoids item()
            actions.append (action)
            logprobs.append (logprob)
        actions = torch.cat (actions)
        logprobs = torch.cat (logprobs)
        '''
 
        
        actions = torch.multinomial(probs, num_samples=self.num_flips, replacement=True) #[num_flips]
        logprobs = torch.log (probs[actions]) #[num_flips]


        
        return actions, logprobs
    
    @classmethod
    def computeReward (cls, actions):
        if actions [-1] == 0 and actions[-2] == 0:
            return 1
        else:
            return 0
    


if __name__ == "__main__":
    init_weight = 0.5
    num_flips = 3

    torch.manual_seed (42)

    policy = CoinPolicy (init_weight, num_flips)
    optimizer = torch.optim.SGD (policy.parameters(), lr = 1e-2)

    epochs = 3000

    thetas = []
    for epoch_num in range(epochs):
        print (f"epoch num = {epoch_num}")

        thetas.append (torch.sigmoid(policy.theta).item())
        print (f" theta = {policy.theta.item()}")


        optimizer.zero_grad()
        actions, logprobs = policy()
        #print (f"actions = {actions}")
        #print (f"log probs = {logprobs}")

        reward = CoinPolicy.computeReward (actions)
        #print (f"reward = {reward}")

        loss = -torch.sum(logprobs) * reward
        #print (f"loss =. {loss.item()}")
        print ("--------------------"*3)

        loss.backward()
        optimizer.step()

    import matplotlib.pyplot as plt
    plt.plot (thetas); plt.xlabel("Epoch");plt.ylabel("p(head)");plt.grid()
    plt.show()

