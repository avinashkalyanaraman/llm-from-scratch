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
    def __init__ (self, module, bucket_size_MB):

        super().__init__()

        self.module = module
        self.my_rank = dist.get_rank()
        self.max_bucket_size = bucket_size_MB * 1024 * 1024 #converting to bytes
        
        #Current Running Bucket Details
        self.constituent_tensors = []
        self.current_bucket_size = 0

        self.work_handles = [] #Stores the handles of the async commxn tasks [per-process]

        #Broadcast rank-0 state to all nodes!
        with torch.no_grad(): #See naive_ddp/run.py for more details on why torch.no_grad() is used!
            for n, p in self.module.named_parameters ():
                dist.broadcast (p, src = 0, async_op = False)

        #Register callback to be invoked when gradient is computed for a parameter
        for p in self.module.parameters():
            p.register_post_accumulate_grad_hook (self.post_grad_compn_hook)

    #Flushes the current bucket and resets stats
    def flushBucket (self):

        for tensor in self.constituent_tensors:
            dist.all_reduce (tensor, op = dist.ReduceOp.SUM, async_op = False)
            tensor.div_(dist.get_world_size())

        #Reset bucket stats!
        self.current_bucket_size = 0 
        self.constituent_tensors = []
        
    def addToBucket (self, tensor):
        self.constituent_tensors.append (tensor)
        self.current_bucket_size += tensor.nbytes

    #callback when gradient is computed for a parameter!
    def post_grad_compn_hook (self, param):

        current_grad_tensor = param.grad
        current_grad_tensor_size = current_grad_tensor.nbytes

        #Flush current bucket and this tensor if it is bigger than a bucket!
        if current_grad_tensor_size > self.max_bucket_size:
            self.flushBucket () 
            
            #Flush this grad tensor. Add to bucket and flush it
            self.addToBucket (current_grad_tensor)
            self.flushBucket()
            

        #Does the addn of this tensor exceed max bucket size. if so flush existing bucket!
        if (current_grad_tensor_size + self.current_bucket_size) > self.max_bucket_size:
            self.flushBucket ()

        #add the current grad tensor to the filling bucket
        self.addToBucket (current_grad_tensor)
        

        
    def forward (self, x):
        return self.module (x)
    
    def finish_gradient_synchronization (self):
        
        
        #Flush any remaining gradient tensors
        self.flushBucket ()
        return
        
        '''
        for work_handle in self.work_handles:
            work_handle.wait()
        for param in self.module.parameters():
            param.grad.div_(dist.get_world_size()) #Divide the accumulated gradient!

        self.work_handles.clear() #Reset the waiting list to be empty for next batch
        '''