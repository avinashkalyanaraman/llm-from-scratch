import torch
import torch.distributed as dist

'''
This class wraps an arbitrary pytorch nn.Module.
It handles 
    (i) Broadcasting rank 0 weights upon init
    (ii) Registering the callback to be invoked when gradients are computed

    (iii) Does the forward pass 
    
    (iv) Before the optimizer.step() can be invoked, the finish_gradient_synchronization() ensures 
        all gradients are in sync. It does so by flattening the gradient tensors of all parameters, 
        and all_reducing them, and reassigning that to the gradient tensors


Note that every process (rank) gets a DDP, but only rank 0 broadcasts
'''

class DDP (torch.nn.Module):
    def __init__ (self, module):

        super().__init__()

        self.module = module
        self.my_rank = dist.get_rank()

        #Broadcast rank-0 state to all nodes!
        with torch.no_grad(): #See naive_ddp/run.py for more details on why torch.no_grad() is used!
            for n, p in self.module.named_parameters ():
                dist.broadcast (p, src = 0, async_op = False)
        
        
    def forward (self, x):
        return self.module (x)
    
    def finish_gradient_synchronization (self):

        agg_grad_tensors = [ele.grad for ele in self.module.parameters()] #list having [param1_grad_tensor, param2_grad_tensor, ...]
        flattened_agg_grad_tensor = torch._utils._flatten_dense_tensors (agg_grad_tensors) #1D tensor having all the contents of above
        dist.all_reduce (flattened_agg_grad_tensor, op = dist.ReduceOp.SUM) #AVG works directly. but no 'gloo' support
        flattened_agg_grad_tensor.div_(dist.get_world_size()) #in-place

        #Let us unflatten it and set it to the gradients .
        # For that we need to know the shapes from a reference tensor list : [agg_grad_tensors]
        #It is just assigned to the same list as lhs value
        agg_grad_tensors = torch._utils._unflatten_dense_tensors (flattened_agg_grad_tensor, agg_grad_tensors)
        for p, acc_grad_tensor in zip (self.module.parameters (), agg_grad_tensors):
            p.grad = acc_grad_tensor
        