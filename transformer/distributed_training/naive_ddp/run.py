import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

def dist_demo (rank, world_size, device):
    init (rank, world_size) #Each worker now knows about others!

    try:
        nn = NN_toy (5, 3, 10, device)

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
        for p in nn.parameters():
            dist.broadcast(p, src = 0, async_op=False)

        #print (f"data in rank {rank} after all-reduce = {my_data[0]}")
        for p in nn.named_parameters ():
            print (f"nn.named parameter in rank {rank} = {p}")
    
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 3

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

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
   
    print("Train dataset size:", len(train_dataset))

    my_nn = NN_toy (384, 768, 10)
    my_nn2 = NN_toy (384, 768, 10)

    my_nn.load_state_dict (my_nn2.state_dict())


    for p1, p2 in zip(my_nn.parameters(), my_nn2.parameters()):
        assert torch.equal(p1, p2)
        print (f"{p1.shape}")


    mp.spawn (fn = dist_demo, args = (world_size,device,), nprocs = world_size, join=True)