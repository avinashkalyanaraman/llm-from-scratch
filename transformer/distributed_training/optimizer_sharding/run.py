import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import copy

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler, Subset

from sharded_optimizer import MyShardedOptimizer

import sys
sys.path.append("../bucketed_overlap_commnxn_compn")
from ddp_overlap_bucketed import DDP


'''
# This is a modular version of the run.py [using wrapper DDP class from ddp_overlap_indiv_params.py]
# code is similar to naive DDP except that
# it leverages the intuition that during backprop,
# gradient for the parameters are computed sequentially (one layer at a time) from the back.
# So as soon as the gradient is computed, all_reduce is done for that gradient. In parallel, the previous layers' 
# gradient gets computed.

#It also does optimizer sharding
'''

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


def init (rank, world_size):
    #Each worker connects with a master and exchanges their information (i.e., how they can be reached!)
    #This helps other workers discover one another, without going via the master!
    #After this, collectives (e.g., all_reduce) run peer-to-peer using Gloo's chosen topology
    #(ring/tree/etc.), not through the master.

    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "25131"
    dist.init_process_group ("gloo", rank = rank, world_size=world_size)

def dist_demo (rank, world_size, train_dataset, num_epochs, worker_batch_size, bucket_size_MB, device):

    torch.manual_seed (42)

    torch.set_default_dtype(torch.float64) #this just helps compare the algorithms and not let adamw "drift" soon when set to float64
    init (rank, world_size) #Each worker now knows about others!

    try:
        nn = NN_toy (784, 1024, 10, device)
        #Wrapper DDP
        ddp_model = DDP (nn, bucket_size_MB)

        #Rank 0 compares weights of no ddp!
        if rank == 0:
            nn_no_ddp = copy.deepcopy (nn)

            
        #Now the models of the workers are all in sync!
        #Each worker deals with its own section of the data. Use DistributedSampler for it
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        ) #Setting shuffle=False to verify correctness of ddp!

        loader = DataLoader(
            train_dataset,
            batch_size= worker_batch_size,
            sampler=sampler,
            shuffle=False 
        )

        #if adamW use float64 to not let drift show up sooner.
        #optimizer = torch.optim.AdamW ( nn.parameters(), lr = 1e-3, betas = (0.9, 0.999), eps=1e-8, weight_decay = 1e-2)
        #optimizer = torch.optim.SGD (nn.parameters(), lr = 1e-3)
        optimizer = MyShardedOptimizer (nn.parameters(), 
                                        torch.optim.AdamW, lr = 1e-3, 
                                        betas = (0.9, 0.999), eps=1e-8, weight_decay = 1e-2)
        criterion = torch.nn.CrossEntropyLoss()
        print (f"in rank {rank} # of batches = {len(loader)}")


        for epoch in range (num_epochs):
            print (f"Running epoch {epoch} in rank {rank}")
            for batchnum, (X, Y) in enumerate (loader):
                X = X.to (device).to(torch.float64) #[worker_batch_size,1, 28, 28]
                Y = Y.to (device) #[worker_batch_size,]

                optimizer.zero_grad (set_to_none= True)

                #Modifying X in the format the NN seeks
                X = X.squeeze(1) #[worker_batch_size,28,28]
                X = X.reshape(X.shape[0], X.shape[-1]* X.shape[-2]) #[worker_batch_size, 784]
               
                #Pass it through the model
                logits = ddp_model(X)

                #Compute loss!
                loss = criterion (logits, Y)
                #print (f"loss in epoch {epoch} in batchnum {batchnum} in rank {rank} = {loss.item()}")

                loss.backward()

                #Ensure all gradients in the workers are in sync
                ddp_model.finish_gradient_synchronization()

                optimizer.step()

        
        #Verify overlapped_ddp in process 0!        
        if rank == 0:
            print (f"Verifying ddp on rank = {rank}")
            loader_standalone = DataLoader(
                train_dataset,
                batch_size= worker_batch_size * world_size,
                shuffle=False 
            )

            optimizer_noddp = torch.optim.AdamW (nn_no_ddp.parameters(), lr = 1e-3, betas = (0.9, 0.999), eps=1e-8, weight_decay = 1e-2)
            #optimizer_noddp = torch.optim.SGD (nn_no_ddp.parameters(), lr = 1e-3) 
            for epoch in range (num_epochs):
                for batchnum, (X,Y) in enumerate (loader_standalone):
                    X = X.to (device).to(torch.float64) #[worker_batch_size,1, 28, 28]
                    Y = Y.to (device) #[worker_batch_size]


                    optimizer_noddp.zero_grad (set_to_none= True)

                    #Modifying X in the format the NN seeks
                    X = X.squeeze(1) #[worker_batch_size*world_size,28,28]
                    X = X.reshape(X.shape[0], X.shape[-1]* X.shape[-2]) #[worker_batch_size*world_size, 784]

                    #Pass it through the model
                    logits = nn_no_ddp(X)

                    #Compute loss!
                    loss = criterion (logits, Y)
                    #print (f"loss in epoch {epoch} in batchnum {batchnum} in rank {rank} = {loss.item()}")

                    loss.backward()
                    optimizer_noddp.step()
                    
                    
            
            #Now we compare weights of nn_noddp and nn (with ddp) on rank0
            for p1, p2 in zip(nn.parameters(), nn_no_ddp.parameters()):
                assert torch.allclose(p1, p2, atol=1e-5, rtol=1e-4)
                

    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    world_size = 6
    num_epochs = 5
    worker_batch_size = 32

    bucket_size_MB = 10


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Loading MNIST data with standard MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts to [0,1] tensor
        transforms.Normalize((0.1307,), (0.3081,))  # mean/std of MNIST
    ])

    # Download + load training set
    train_dataset = datasets.MNIST(
        root="../naive_ddp/data",
        train=True,
        download=True,
        transform=transform
    )
    #just for comparison reasons with no-ddp. we avoid stray batches in a worker
    train_dataset = Subset(train_dataset, range(worker_batch_size * world_size * 5)) 
   
    print("Train dataset size:", len(train_dataset))


    mp.spawn (fn = dist_demo, args = (world_size,train_dataset, num_epochs, worker_batch_size, bucket_size_MB,device,),
               nprocs = world_size, join=True)