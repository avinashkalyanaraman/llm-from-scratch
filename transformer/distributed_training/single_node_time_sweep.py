import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import defaultdict
import plot_utils


def init (rank, world_size):
    #Each worker connects with a master and exchanges their information (i.e., how they can be reached!)
    #This helps other workers discover one another, without going via the master!
    #After this, collectives (e.g., all_reduce) run peer-to-peer using Gloo's chosen topology
    #(ring/tree/etc.), not through the master.

    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "25131"
    dist.init_process_group ("gloo", rank = rank, world_size=world_size)

def dist_demo (rank, world_size, tensor_size):
    init (rank, world_size)

    try:
        data = torch.randn (tensor_size, dtype = torch.float32)
        #print (f"data in rank {rank} before all-reduce = {data}")

        work = dist.all_reduce (data, async_op=True) #non-blocking. so free to do anything now
        #do any compute here to mask compute and communication
        #if you don't want to do any overlap, just call async_op=False & remove the wait() below
        work.wait() #to ensure that the collection finishes

        #print (f"data in rank {rank} after all-reduce = {data}")
    
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    
    world_sizes = [2, 4, 6]
    tensor_sizes = [2**18, 10 * 2**18, 100 * 2**18, 2 ** 28] #[1MB, 10MB, 100MB, 1GB] of float32s

    iterations_to_ignore = 3
    num_runs = 2

    timings = defaultdict (list)

    for world_size in world_sizes:
        print (f'Handling world size = {world_size}')
        
        for tensor_size in tensor_sizes:

            print (f"Handling tensor size = {tensor_size}")

            for run_num in range(iterations_to_ignore + num_runs):
                start = time.perf_counter()
                mp.spawn (fn = dist_demo, args = (world_size,tensor_size), nprocs = world_size, join=True)
                end = time.perf_counter()
                duration = end - start
                if run_num < iterations_to_ignore : 
                    continue

                timings [(world_size, tensor_size)].append (duration)

                
        
    avg_timings = {k: round (sum(v) / len(v), 2) for k, v in timings.items()}
    print (f"Avg Timings = {avg_timings}")
    plot_utils.grouped_bar_plot (avg_timings)