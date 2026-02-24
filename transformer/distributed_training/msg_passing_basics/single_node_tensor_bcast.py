import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp



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
        #Broadcast parameters from rank = 0 to all nodes!

        data = torch.randint (0, 10, (3,))
        data2 = torch.randint (0, 10, (3,))
        dict_data = {'data1' : data}

        #my_data = [dict_data, data2] #Sending two objects
        my_data = [dict_data, data2] if rank == 0 else [None, None] #better way to write v/s the above since non broadcaster are going to be overwritten!
        print (f"data in rank {rank} before broadcast = {my_data}")

        dist.broadcast_object_list (my_data, src = 0) #broadcast will block on all nodes!

        print (f"data in rank {rank} after all-reduce = {my_data}")

        #Broadcasting just tensors!
        print (f"tensor in rank {rank} before broadcast = {data}")
        dist.broadcast (data, src = 0, async_op= False)
        print (f"tensor in rank {rank} after broadcast = {data}")
        
    
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    mp.spawn (fn = dist_demo, args = (world_size,device,), nprocs = world_size, join=True)