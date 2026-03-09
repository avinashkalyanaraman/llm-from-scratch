import argparse
import sys, os
sys.path.append("../")

import torch.distributed as dist
import torch.multiprocessing as mp

import torch
import statistics

from transformer_pipeline import Transformer
import optimizer

import timeit

from distributed_training.naive_ddp.ddp import DDP as DDP_NAIVE
from distributed_training.flattened_ddp.ddp import DDP as DDP_FLATTENED
from distributed_training.overlap_commnxn_compn.ddp_overlap_indiv_params import DDP as DDP_OVERLAP
from distributed_training.bucketed_overlap_commnxn_compn.ddp_overlap_bucketed import DDP as DDP_BUCKETEDOVERLAP

torch.manual_seed(42)

def getCrossEntropyRHS (q):
    
    max_q, _ = torch.max (q, dim=-1, keepdim=True) #[batchsize, seqlen, 1]
    q = q - max_q #[batch, seqlen, vocabsize] - [batch, seqlen, 1] --> broadcased to [batch, seqlen, vocabsize]

    exp_x = torch.exp (q)
    sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True) #[batch, seqlen, 1]

    rhs = q - torch.log(sum_exp_x)

    return rhs

def getCrossEntropyLossFromClass (p,q):
    rhs = getCrossEntropyRHS(q)

    #Now we need to do the equivalrnt of:
    #for i in Batch:
    # for j in seqlen:
    #   r[i,j] = q[i,j, p[i,j]]

    #To do the above we can gather:
    rhs = torch.gather(rhs, dim=-1, index=p.unsqueeze(-1)) #[batchlen, seqlen, 1]
    rhs = rhs.squeeze(-1)
    all_loss = -rhs
    return torch.mean(all_loss)


def benchmark (func, args=None, device=None):
    start = timeit.default_timer()
    retval = func(args)
    torch.cuda.synchronize(device) 
    end = timeit.default_timer()
    runtime = end-start
    return retval, runtime

def init (rank, world_size):
    #Each worker connects with a master and exchanges their information (i.e., how they can be reached!)
    #This helps other workers discover one another, without going via the master!
    #After this, collectives (e.g., all_reduce) run peer-to-peer using NCCL's chosen topology
    #(ring/tree/etc.), not through the master.

    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "25131"
    dist.init_process_group ("nccl", rank = rank, world_size=world_size)

def dist_benchmarking (rank, world_size, lr, beta1, beta2,d_model, seqlen, heads, 
                       worker_batchsize, num_layers,isTorchCompile, epochs, warmups,
                        vocab_size, ddp_type):

    torch.manual_seed (42)

    init (rank, world_size) #Each worker now knows about others!

    try:
        print (f" in rank {rank}")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device (rank) #Sets the default (cuda) device for the current process


        #Create a random set of input and output for each process spawned
        sample_input = torch.randint ( 0, vocab_size, (worker_batchsize, seqlen), device = device)
        sample_output = torch.randint(0, vocab_size, (worker_batchsize, seqlen), device=device)


        model = Transformer (d_model=d_model, num_heads=heads, d_ff=None, 
                            vocab_size=vocab_size, num_layers= num_layers,
                            max_seq_len=seqlen, device=device, dtype=torch.float32 )
        optim = optimizer.AdamW ( model.parameters(), lr = lr, 
                                 betas = (beta1,beta2), eps=1e-8, weight_decay = 1e-2)


        model.to(device)

        if rank == 0:
            total_num_trainable_params= sum([ele.numel() for ele in model.parameters() if ele.requires_grad])
            print (f"total # trainable params in {ddp_type} model = {total_num_trainable_params/1e6}M")


        #Torch-compile:
        if isTorchCompile:
            model = torch.compile(model)

        fw_runtimes = []
        bp_runtimes = []

        if ddp_type == 'naive':
            ddp_model = DDP_NAIVE (model)
        elif ddp_type == 'flattened':
            ddp_model = DDP_FLATTENED (model)
        elif ddp_type == 'overlap':
            ddp_model = DDP_OVERLAP (model)
        elif ddp_type == 'bucketedoverlap':
            ddp_model = DDP_BUCKETEDOVERLAP (model)
        else:
            assert False , "Incorrect ddp_type; one of [naive, flattened, overlap, bucketedoverlap]"


        for epoch in range(epochs):

            #zero-out gradient
            optim.zero_grad(set_to_none = True)

            #Run forward pass
            y_hat, fw_runtime = benchmark (ddp_model, args = sample_input, device=device)
            #print (f"{y_hat.shape}")

            loss = getCrossEntropyLossFromClass (sample_output, y_hat)

            #Run backprop!
            _, bp_runtime = benchmark (loss.backward, args = None, device=device)

            #print (f"loss after epoch {epoch} = {loss.item()}")

            if epoch >= warmups:
                fw_runtimes.append (fw_runtime)
                bp_runtimes.append (bp_runtime)
                print (f"Rank : {rank} -- Fwd time = {fw_runtime}")
                print (f"Rank : {rank} -- Backprop time = {bp_runtime}")


            #Update gradient!
            optim.step()


        print (f"Rank : {rank} -- Mean fw pass after warmup of {warmups} steps= {statistics.mean(fw_runtimes)} s")
        print (f"Rank : {rank} -- stdev fw pass after warmup of {warmups} steps= {statistics.stdev(fw_runtimes)} s")
        print (f"Rank : {rank} -- Mean bw pass after warmup of {warmups} steps= {statistics.mean(bp_runtimes)} s")
        print (f"Rank : {rank} -- stdev bw pass after warmup of {warmups} steps= {statistics.stdev(bp_runtimes)} s")



    finally:
        dist.destroy_process_group()

if __name__ == '__main__' :

    #Get a list of hyper-params as CLI
    #Learning rate, beta1, beta2, tokenizer, d_model, context-len, dff, n-heads, p-enc, 
    # Create the parser
    parser = argparse.ArgumentParser(description="Transformer DDP-benchmark Run")
    parser.add_argument("--lr", default = 1e-3, type=float, help="Learning Rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for adamw")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for adamw")
    parser.add_argument("--d_model", type=int, default=384, help="dmodel for transformer")
    parser.add_argument("--seqlen", type=int, default=1024, help="context length")
    parser.add_argument("--heads", type=int, default=12, help="number of heads")
    parser.add_argument("--batchsize", type=int, default=1, help="batchsize")
    parser.add_argument("--num_layers", type=int, default=1, help="num of layers")
    parser.add_argument("--epochs", type=int, default=1, help="num of epochs")
    parser.add_argument("--warmups", type=int, default=1, help="num of warmup epochs before timing!")
    parser.add_argument("--tcompile", type=bool, default=False, help="torch compile?")
    parser.add_argument("--vocabsize", type=int, default=50304, help="vocab size of the tokenizer used!")
    parser.add_argument("--num_workers", type=int, default=1, help="# of workers (world size)")
    parser.add_argument("--type", type=str, default="bucketedoverlap", help="[naive, flattened, overlap, bucketedoverlap]")


    args = parser.parse_args()

    #the local vars
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    d_model = args.d_model
    seqlen = args.seqlen
    heads = args.heads
    batchsize = args.batchsize
    num_layers = args.num_layers
    isTorchCompile = args.tcompile
    epochs = args.epochs
    warmups = args.warmups
    vocab_size = args.vocabsize
    ddp_type = args.type
    num_workers = args.num_workers

    assert batchsize % num_workers == 0 , "Batchsize should be equally divisible by # workers"
    worker_batchsize = batchsize // num_workers


    #set the device!
    if not torch.cuda.is_available():
        assert False , "Cuda required for this benchmark test"
    

    #Once we know the parameters, let us create the workers!
    mp.spawn (fn = dist_benchmarking, args = (num_workers,lr, beta1, beta2,
                                              d_model, seqlen, heads, worker_batchsize,
                                               num_layers,isTorchCompile, epochs,
                                                warmups, vocab_size, ddp_type
                                                ), nprocs = num_workers, join=True)
    
    