import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import copy

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler, Subset

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
    os.environ["MASTER_PORT"] = "25133"
    dist.init_process_group ("gloo", rank = rank, world_size=world_size)

def dist_demo (rank, world_size, train_dataset, num_epochs, worker_batch_size, device):

    torch.manual_seed (42)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


    init (rank, world_size) #Each worker now knows about others!

    try:
        nn = NN_toy (784, 1024, 10, device)

        #Rank 0 compares weights of no ddp!
        if rank == 0:
            nn_no_ddp = copy.deepcopy (nn)

        '''
        #Type 1:        
        #Broadcast parameters from rank = 0 to all nodes!

        my_state_dict = nn.state_dict()
        my_data = [my_state_dict] #To broadcast the state-dict, wrap it into a list. non broadcaster's my_data will get overwritten
        dist.broadcast_object_list (my_data, src = 0) #broadcast will block on all nodes, and the my_data list on all nodes now is identical!
        if rank != 0:
            nn.load_state_dict(my_data[0])
        '''
        
        #Alternate way since Parameters is a subclass of Tensor. Can also broadcast as below!
        with torch.no_grad(): #Without this: you will see a silent incorrect behavior warning 
            # dist.broadcast performs an in-place write to Parameter tensors.
            # Since Parameters require grad, we prevent autograd from tracking this non-differentiable operation.
            # In the earlier case, load_state_dict is not treated as a grad op.
            for p in nn.parameters():
                dist.broadcast(p, src = 0, async_op=False)

        #print (f"data in rank {rank} after all-reduce = {my_data[0]}")
        for p in nn.named_parameters ():
            print (f"nn.named parameter in rank {rank} = {p}")

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

        optimizer = torch.optim.AdamW ( nn.parameters(), lr = 1e-3, betas = (0.9, 0.999), eps=1e-8, weight_decay = 1e-2, foreach=False)
        #optimizer = torch.optim.SGD (nn.parameters(), lr = 1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        print (f"in rank {rank} # of batches = {len(loader)}")


        for epoch in range (num_epochs):
            print (f"Running epoch {epoch} in rank {rank}")
            for batchnum, (X, Y) in enumerate (loader):
                X = X.to (device) #[worker_batch_size,1, 28, 28]
                Y = Y.to (device) #[worker_batch_size,]


                optimizer.zero_grad (set_to_none= True)

                #Modifying X in the format the NN seeks
                X = X.squeeze(1) #[worker_batch_size,28,28]
                X = X.reshape(X.shape[0], X.shape[-1]* X.shape[-2]) #[worker_batch_size, 784]

                #Pass it through the model
                logits = nn(X)

                #Compute loss!
                loss = criterion (logits, Y)
                #print (f"loss in epoch {epoch} in batchnum {batchnum} in rank {rank} = {loss.item()}")

                loss.backward()

                with torch.no_grad(): #unnecessary since p.grad has requires_grad = False
                    for n, p in nn.named_parameters():
                        dist.all_reduce (p.grad, op = dist.ReduceOp.SUM) #AVG works directly. but no 'gloo' support
                        p.grad.div_(world_size) #in-place

                        #if epoch == num_epochs - 1:
                            #print (f"p.grad for param = {n} in rank {rank} in epoch {epoch}= {p.grad}")

                optimizer.step()
                if rank == 0 and epoch == 0 and batchnum == 0:
                    ddp_grads = {n: p.grad.detach().cpu().clone() for n, p in nn.named_parameters()}

            
        #Verify ddp in process 0!        
        if rank == 0:
            print (f"Verifying ddp on rank = {rank}")
            loader_standalone = DataLoader(
                train_dataset,
                batch_size= worker_batch_size * world_size,
                shuffle=False 
            )

            optimizer_noddp = torch.optim.AdamW (nn_no_ddp.parameters(), lr = 1e-3, betas = (0.9, 0.999), eps=1e-8, weight_decay = 1e-2, foreach=False)
            #optimizer_noddp = torch.optim.SGD (nn_no_ddp.parameters(), lr = 1e-3) 
            for epoch in range (num_epochs):
                for batchnum, (X,Y) in enumerate (loader_standalone):
                    X = X.to (device) #[worker_batch_size,1, 28, 28]
                    Y = Y.to (device) #[worker_batch_size]


                    optimizer_noddp.zero_grad (set_to_none= True)

                    #Modifying X in the format the NN seeks
                    X = X.squeeze(1) #[worker_batch_size*world_size,28,28]
                    X = X.reshape(X.shape[0], X.shape[-1]* X.shape[-2]) #[worker_batch_size*world_size, 784]

                    #Let us get "worker" granularity chunks
                    X_chunks = X.chunk(world_size, dim=0) #tuple of world_size ([worker_batch_size, 784], [worker_batch_size, 784]...)
                    Y_chunks = Y.chunk(world_size, dim=0) #tuple of world_size ([worker_batch_size,], [worker_batch_size,] ...)

                    for i in range(world_size):
                        logits = nn_no_ddp(X_chunks[i])
                        loss_i = criterion(logits, Y_chunks[i]) # mean over one batch
                        (loss_i / world_size).backward() # avg across ranks

                    optimizer_noddp.step()

                    '''
                    #Pass it through the model
                    logits = nn_no_ddp(X)

                    #Compute loss!
                    loss = criterion (logits, Y)
                    #print (f"loss in epoch {epoch} in batchnum {batchnum} in rank {rank} = {loss.item()}")

                    loss.backward()
                    optimizer_noddp.step()
                    '''

                    if epoch == 0 and batchnum == 0:
                        base_grads = {n: p.grad.detach().cpu().clone() for n, p in nn_no_ddp.named_parameters()}
        
            #Now we compare weights of nn_noddp and nn (with ddp) on rank0
            #for p1, p2 in zip(nn.parameters(), nn_no_ddp.parameters()):
            #    assert torch.allclose(p1, p2, atol=1e-5, rtol=1e-4)

            for n in ddp_grads:
                diff = (ddp_grads[n] - base_grads[n]).abs().max().item()
                print("grad diff", n, diff)
            
            sd1 = nn.state_dict()
            sd2 = nn_no_ddp.state_dict()
            for k in sd1:
                if not torch.allclose(sd1[k], sd2[k], atol=1e-8, rtol=0):
                    print("mismatch", k, (sd1[k] - sd2[k]).abs().max().item())
                    break
            
            #Now we compare weights of nn_noddp and nn (with ddp) on rank0
            for p1, p2 in zip(nn.parameters(), nn_no_ddp.parameters()):
                assert torch.allclose(p1, p2, atol=1e-5, rtol=1e-4)

                

                    
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 8
    num_epochs = 20
    worker_batch_size = 256



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Loading MNIST data with standard MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts to [0,1] tensor
        transforms.Normalize((0.1307,), (0.3081,))  # mean/std of MNIST
    ])

    # Download + load training set
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    #just for comparison reasons with no-ddp. we avoid stray batches in a worker
    train_dataset = Subset(train_dataset, range(worker_batch_size * world_size * 5)) 
   
    print("Train dataset size:", len(train_dataset))


    mp.spawn (fn = dist_demo, args = (world_size,train_dataset, num_epochs, worker_batch_size, device,), nprocs = world_size, join=True)