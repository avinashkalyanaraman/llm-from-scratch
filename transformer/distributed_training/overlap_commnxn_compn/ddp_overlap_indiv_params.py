import torch
import torch.distributed as dist

'''
This class wraps an arbitrary pytorch nn.Module.
It handles 
    (i) Broadcasting rank 0 weights upon init
    (ii) Registering the callback to be invoked when gradients are computed

    (iii) Does the forward pass 
    (iv) Handles the callback that gets invoked when loss.backward() is invoked!
            The callback asynchronously sends the gradient of param just computed to other nodes
    
    (v) Before the optimizer.step() can be invoked, the finish_gradient_synchronization() ensures 
        all gradients are in sync. It waits for the async processes to complete, and divides by
        the total number of ranks to account for the gradient accumulation.


Note that every process (rank) gets a DDP, but only rank 0 broadcasts
'''

class DDP (torch.nn.Module):
    def __init__ (self, module):

        super().__init__()

        self.module = module
        self.my_rank = dist.get_rank()

        self.work_handles = [] #Stores the handles of the async commxn tasks [per-process]

        #Broadcast rank-0 state to all nodes!
        with torch.no_grad(): #See naive_ddp/run.py for more details on why torch.no_grad() is used!
            for n, p in self.module.named_parameters ():
                dist.broadcast (p, src = 0, async_op = False)

        #Register callback to be invoked when gradient is computed for a parameter
        for p in self.module.parameters():
            p.register_post_accumulate_grad_hook (self.post_grad_compn_hook)

    #callback when gradient is computed for a parameter!
    def post_grad_compn_hook (self, param):

        #Here we can do an all_reduce on the gradient.
        #While that is being done, we can go ahead and compute the gradient on the "prev" layer of the NN
        with torch.no_grad():
            handle = dist.all_reduce (param.grad, async_op= True, op = dist.ReduceOp.SUM)

        #Note that after this operation, all gradient values still need to be divided by number of workers!
        self.work_handles.append (handle)
        #print (f"len of work handles in rank {dist.get_rank()} is {len(work_handles)}")
    
    def forward (self, x):
        return self.module (x)
    
    def finish_gradient_synchronization (self):
        for work_handle in self.work_handles:
            work_handle.wait()
        for param in self.module.parameters():
            param.grad.div_(dist.get_world_size()) #Divide the accumulated gradient!

        self.work_handles.clear() #Reset the waiting list to be empty for next batch