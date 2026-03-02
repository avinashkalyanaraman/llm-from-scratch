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

        #List of tensors of each bucket.
        # This tracks the grad tensors, and 
        # aids in copying back to the grad-tensors after asynchrnous all-reduce

        self.all_buckets = [] #[ [tensor1 of bucket1, tensor2 of bucket1...],
                                # [tensor1 of bucket2, tensor2 of bucket2...],
                                # [tensor1 of bucket3, tensor2 of bucket3...] ]
        self.all_transfers = [] #[ [flattened list of bucket1],
                                #   [ flattened list of bucket2] ...]

        self.work_handles = [] #Stores the handles of the async commxn tasks [per-process]

        #Broadcast rank-0 state to all nodes!
        with torch.no_grad(): #See naive_ddp/run.py for more details on why torch.no_grad() is used!
            for n, p in self.module.named_parameters ():
                dist.broadcast (p, src = 0, async_op = False)

        #Register callback to be invoked when gradient is computed for a parameter
        for p in self.module.parameters():
            p.register_post_accumulate_grad_hook (self.post_grad_compn_hook)

    #Flushes the current bucket and resets stats. Syncrhonously done.
    #Just for book-keeping purposes. Unused.
    def flushBucketSync (self):    
        
        #1D tensor having all the contents of grad tensors of this bucket flattened
        flattened_agg_grad_tensor = torch._utils._flatten_dense_tensors (self.constituent_tensors) 
        dist.all_reduce (flattened_agg_grad_tensor, op = dist.ReduceOp.SUM) #AVG works directly. but no 'gloo' support
        flattened_agg_grad_tensor.div_(dist.get_world_size()) #in-place

        self.new_constituent_tensors = torch._utils._unflatten_dense_tensors (flattened_agg_grad_tensor, 
                                                                          self.constituent_tensors)
        for old, new in zip (self.constituent_tensors, self.new_constituent_tensors):
            old.copy_(new)

        #Reset bucket stats!
        self.current_bucket_size = 0 
        self.constituent_tensors = []
    
    #The async method. Flushes the current bucket and resets stats
    def flushBucket (self):    
        
        #All flushed/nothing to flush!
        if len(self.constituent_tensors) == 0:
            return

        #Adding the list of current tensors to the bucket tracking list!
        self.all_buckets.append (self.constituent_tensors)

        #1D tensor having all the contents of grad tensors of this bucket flattened
        flattened_agg_grad_tensor = torch._utils._flatten_dense_tensors (self.constituent_tensors)
        self.all_transfers.append (flattened_agg_grad_tensor) #list of lists

        work = dist.all_reduce (flattened_agg_grad_tensor, op = dist.ReduceOp.SUM, async_op=True) #AVG works directly. but no 'gloo' support
        self.work_handles.append (work)


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
    
        for work_handle, bucket, flattened_xfer in zip(self.work_handles, self.all_buckets, self.all_transfers):
            work_handle.wait()

            #The given async transfer is complete!
            flattened_xfer.div_(dist.get_world_size())

            new_constituent_tensors = torch._utils._unflatten_dense_tensors (flattened_xfer, 
                                                                          bucket)
            #each element of bucket basically has a reference to a param.grad tensor
            for old, new in zip (bucket, new_constituent_tensors):
                old.copy_(new)

        self.work_handles.clear() #Reset the waiting list to be empty for next batch
        self.all_transfers.clear() #Reset buckets and transfers for next batch
        self.all_buckets.clear() 
        