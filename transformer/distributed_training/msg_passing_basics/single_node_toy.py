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

def dist_demo (rank, world_size):
    init (rank, world_size)

    try:
        data = torch.randint (0, 10, (3,))
        print (f"data in rank {rank} before all-reduce = {data}")

        work = dist.all_reduce (data, async_op=True, op=dist.ReduceOp.SUM) #non-blocking. so free to do anything now. default reduction op=SUM
        #do any compute here to mask compute and communication
        #if you don't want to do any overlap, just call async_op=False & remove the wait() below
        work.wait() #to ensure that the collection finishes

        print (f"data in rank {rank} after all-reduce = {data}")
    
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 3
    mp.spawn (fn = dist_demo, args = (world_size,), nprocs = world_size, join=True)