import torch
import torch.distributed as dist

'''
This class wraps an arbitrary pytorch nn.Module.
It handles 
    (i) Broadcasting rank 0 weights upon init
    (ii) Registering the callback to be invoked when gradients are computed

    (iii) Does the forward pass 
    
    (iv) Before the optimizer.step() can be invoked, the finish_gradient_synchronization() ensures 
        all gradients are in sync. It individually does an all reduce on each gradient tensor.


Note that every process (rank) gets a DDP, but only rank 0 broadcasts
'''

class DDP (torch.nn.Module):
    def __init__ (self, module):

        super().__init__()

        self.module = module
        self.my_rank = dist.get_rank()

                #Broadcast parameters from rank = 0 to all nodes!
        '''
        #Type 1:        

        my_state_dict = self.module.state_dict()
        my_data = [my_state_dict] #To broadcast the state-dict, wrap it into a list. non broadcaster's my_data will get overwritten
        dist.broadcast_object_list (my_data, src = 0) #broadcast will block on all nodes, and the my_data list on all nodes now is identical!
        if rank != 0:
            self.module.load_state_dict(my_data[0])
        '''

        #Broadcast rank-0 state to all nodes!
        with torch.no_grad(): #See naive_ddp/run.py for more details on why torch.no_grad() is used!
            for n, p in self.module.named_parameters ():
                dist.broadcast (p, src = 0, async_op = False)
        
        
    def forward (self, x):
        return self.module (x)
    
    def finish_gradient_synchronization (self):

        with torch.no_grad(): #unnecessary since p.grad has requires_grad = False
            for n, p in self.module.named_parameters():
                dist.all_reduce (p.grad, op = dist.ReduceOp.SUM) #AVG works directly. but no 'gloo' support
                p.grad.div_(dist.get_world_size()) #in-place
        