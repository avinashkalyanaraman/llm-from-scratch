import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp


class NN_toy (torch.nn.Module):
    def __init__ (self, dim1, dim2, out, device = None):
        super().__init__()

        self.lin1 = torch.nn.Linear (dim1, dim2, device=device)
        self.relu1 = torch.nn.ReLU ()
        self.lin2 = torch.nn.Linear (dim2, out, device=device)

    def forward (self, x):
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2 (x)
        return x

class MyShardedOptimizer (torch.optim.Optimizer):
    def __init__ (self, params, optimizer_cls, **kwargs):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.global_idx = 0 #Tracks the total number of parameters [added by all]

        #Let us set the optimizer to only track 'parameters' based on rank!
        self.my_params = []


        #Params maybe an iterable of dicts (parameter groups) or an iterable of tensors (= single param group)
        # optimizer.ABC (nn.parameters(), ..defaults..) (or)
        # optimizer.ABC ( [ {'params' : nn1.parameters(), 'lr' : 0.01},
        #                   {'params' : nn2.parameters(), }],
        #                   ..defaults.. )
        
        params = self.normalize (params) #handles the two cases mentioned above!
        defaults = kwargs
        #Now params is an iterable of dicts! 
        # each dict may have hyper-parameters it overrides from the default

        self._all_params = [p for g in params for p in g["params"]]
        
        
        for group in params: 
            #params is an iterable of dicts [{'params': [iterable of param tensor], 
            # 'hyperparam1' : val1, 'hyperparam2' : val2}!
            group_params = list (group ['params'])
            all_other_kvs = {k:v for k, v in  group.items() if k != 'params'}
            
            my_params_in_this_group = []
            for param_num, param in enumerate(group_params):

                #My parameter!
                if (self.global_idx) % self.world_size == self.rank:                    
                    my_params_in_this_group.append (param)
                self.global_idx += 1
            
            this_param_group_to_track = all_other_kvs.copy() #Shallow copy
            this_param_group_to_track ['params'] = my_params_in_this_group
            self.my_params.append (this_param_group_to_track)
            #print (f"rank {self.rank} :: self.my_params = {self.my_params}")

        self.opt = None 
        #initializing to None so that add_param_group can differentiate
        #between the below call, and the user invoking add_param_group()
        #in the training loop!

        super().__init__ (self.my_params, defaults) 
        self.opt = optimizer_cls (self.my_params, **kwargs)


        print (f"rank : {self.rank} - After init total number of params distributed = {self.global_idx}")

        return
    
    @staticmethod
    def normalize (params):

        if isinstance (params, torch.Tensor):
            raise TypeError("params should be an iterable of parameters, not a Tensor")

        params = list(params) #if input is a list, then params remains the same. list([1,2,3]) = [1,2,3]

        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")

        if not isinstance (params[0], dict):
            params = [ {'params': params}]

        return params
    
    def step (self, closure = None):
        loss = self.opt.step (closure) #each rank updates its parameters. 
        #now we need to broadcast our paramters to all!
        #TODO: Implement bucketing like DDP!
        handles = []
        with torch.no_grad():
            for param_num, param in enumerate(self._all_params):
                    wait_handle = dist.broadcast (param, src = param_num % self.world_size, async_op=True)
                    handles.append (wait_handle)
        
        #wait for all the async opreations to finish before returning!
        for handle in handles:
            handle.wait()
        
        return loss

    def add_param_group(self, param_group):

        if self.opt == None:
            return super().add_param_group(param_group) #This sets our param_groups but it is unused!
        
        else: #user-adding param group (i.e., not during init)
            print ('TODO : Not implemented yet!')


    #We want to zero out all the gradients -- not just our parameters'
    # else, next foward pass will exchange incorrect [accumulated] gradients 
    def zero_grad(self, set_to_none = True):
        for p in self._all_params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()


def init (rank, world_size):
    #Each worker connects with a master and exchanges their information (i.e., how they can be reached!)
    #This helps other workers discover one another, without going via the master!
    #After this, collectives (e.g., all_reduce) run peer-to-peer using Gloo's chosen topology
    #(ring/tree/etc.), not through the master.

    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "25131"
    dist.init_process_group ("gloo", rank = rank, world_size=world_size)

def dist_demo (rank, world_size, ):
    torch.manual_seed (42)

    torch.set_default_dtype(torch.float64) #this just helps compare the algorithms and not let adamw "drift" soon when set to float64
    init (rank, world_size) #Each worker now knows about others!

    try :
        nn = NN_toy (4,8, 3)
        #myopt = MyShardedOptimizer (nn.parameters(), torch.optim.AdamW, lr = 1e-3, betas = (0.9, .999),)
        myopt2 = MyShardedOptimizer ([{'params' : list(nn.parameters())[0:2], "lr" : 1e-3},
                               {'params' : list(nn.parameters())[2:], "eps" : 1e-7 }], torch.optim.AdamW, lr = 1e-4)


        
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    world_size = 2
    mp.spawn (fn = dist_demo, args = (world_size,),
               nprocs = world_size, join=True)
    