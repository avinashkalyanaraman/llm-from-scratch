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

       # Each rank has 4 chunks
        input_list = [torch.ones(2, 2) * (rank + i)
                    for i in range(world_size)]

        # Output is one chunk
        output = torch.zeros(2, 2)

        dist.reduce_scatter(
            output,
            input_list,
            op=dist.ReduceOp.SUM
        )

        print(f"Rank {rank} output:\n{output}")

    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    mp.spawn (fn = dist_demo, args = (world_size,device,), nprocs = world_size, join=True)