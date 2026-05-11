import torch
import sys
import argparse

sys.path.append("./../")

import filehandler
from torch.utils.data import DataLoader

from transformer_pipeline import Transformer
import optimizer
import loss

import timeit, statistics

import torch.cuda.nvtx as nvtx


if __name__ == '__main__' :

    #Get a list of hyper-params as CLI
    #Learning rate, beta1, beta2, tokenizer, d_model, context-len, dff, n-heads, p-enc, 
    # Create the parser
    parser = argparse.ArgumentParser(description="Transformer Run")
    parser.add_argument("--lr", default = 1e-3, type=float, help="Learning Rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for adamw")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for adamw")
    parser.add_argument("--d_model", type=int, default=384, help="dmodel for transformer")
    parser.add_argument("--seqlen", type=int, default=128, help="context length")
    parser.add_argument("--heads", type=int, default=12, help="number of heads")
    parser.add_argument("--batchsize", type=int, default=1, help="batchsize")
    parser.add_argument("--num_layers", type=int, default=1, help="num of layers")
    parser.add_argument("--vocabsize", type=int, default=50304, help="vocab size of the tokenizer used! Power of 64 helps!")
    parser.add_argument("--tfile", type=str, default="temp/temp.npy", help="file with tokens to be used as training")
    parser.add_argument("--torchcompile", action="store_true", help="enable torchcompile")
    parser.add_argument("--steps", type=int, default=20, help="total # of steps after warmup to run" )
    parser.add_argument("--stream", action="store_true", help="whether to stream a training batch while compute is happening on gpu!")

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
    vocab_size = args.vocabsize
    t_fname = args.tfile
    isTorchCompile = args.torchcompile
    max_steps = args.steps
    isStream = args.stream

    epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if device.type == "cuda":
        # Enable TF32 tensor cores for FP32 matmuls/convs
        torch.set_float32_matmul_precision("high")
    else:
        assert False, "nvtx is only for cuda device!"


    train_dataset = filehandler.myDataset (t_fname,seqlen)
    print (f"The dataset has  {len(train_dataset)} entires for training")


    # Split the dataset
    train_dataloader = DataLoader (train_dataset, batch_size=batchsize, shuffle=False, drop_last=True, pin_memory=True)
    #pin_memory=True locks the batch page in memory, thus preventing swap out. This permits gpu dma and perf gains! 


    model = Transformer (d_model=d_model, num_heads=heads, d_ff=None, 
                         vocab_size=vocab_size, num_layers= num_layers,
                          max_seq_len=seqlen, device=device, dtype=torch.float32 )
    model.to(device)


    if isTorchCompile:
        model = torch.compile(model)

    opt = optimizer.AdamW ( model.parameters(), lr = lr, betas = (beta1,beta2), eps=1e-8, weight_decay = 1e-2)

    #For cosine annealing:  f(tokenlimit) as opp to f(train_dataloader)
    tokenlimit = 40_000_000
    tc = tokenlimit/(batchsize*seqlen)  #Suggested best practice: is to reach alpha_min at tokenlimit
    tw = 0.05*tc  #warmup fraction is 5% of tc!
    alpha_max = lr
    alpha_min = 1e-6

    num_steps = 0
    tokens_handled = 0

    warmup_steps = 5
    step_times = []
    throughputs = []


    total_num_trainable_params= sum([ele.numel() for ele in model.parameters() if ele.requires_grad])
    print (f"total # trainable params in model = {total_num_trainable_params/1e6}M")

    if not isStream:

        for batchnum, (X,Y) in enumerate (train_dataloader): 

            if num_steps == warmup_steps:
                nvtx.range_push("POST-WARMUP") #Setting this label helps profile post-warmup performance

            start = timeit.default_timer()       

            with nvtx.range(f"h2d-{num_steps}"):
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True) 

            opt.zero_grad(set_to_none=True) #Faster as : sets each parameter’s .grad to None instead of a tensor of zeros

            with nvtx.range(f"forward-{num_steps}"):
                y_hat = model (X)

            with nvtx.range(f"loss-{num_steps}"):
                computed_loss = loss.getCrossEntropyLossFromClass(Y, y_hat)

            #Set learning rate based on schedule!
            with nvtx.range(f"lr-{num_steps}"):
                t = num_steps
                for group in opt.param_groups: #there are a set of param groups
                    curr_lr = optimizer.getCurrentLearningRateBasedOnSchedule (t, alpha_max, alpha_min, tw,tc)
                    group['lr'] = curr_lr

            with nvtx.range(f"backward-{num_steps}"):
                computed_loss.backward() #Computes the gradients
                    
            with nvtx.range(f"clip-{num_steps}"):
                optimizer.gradientClipping (list(model.parameters()), 1) #Clip the gradients
            
        
            with nvtx.range(f"optimizer-{num_steps}"):
                opt.step() #Updates the weights

            tokens_handled += Y.numel()
            num_steps += 1

            #Timing
            end = timeit.default_timer()
            elapsed = end - start
            throughput = Y.numel() / max(elapsed, 1e-9)
            step_times.append (elapsed)
            throughputs.append(throughput)


            #Have we run enough
            if num_steps >= max_steps:
                nvtx.range_pop()            
                break


        print (f"mean running time = {statistics.mean(step_times[warmup_steps:]):0.2f}s")
        print (f"mean throughput = {statistics.mean(throughputs[warmup_steps:]):0.2f} tokens/sec")

    else:
        #the second stream; the copy stream which is used to copy data! 
        # compute kernels go on the default stream!
        copy_stream = torch.cuda.Stream()
        it = iter(train_dataloader)

        # --- preload first batch on the copy stream ---
        host_X, host_Y = next(it)                                # CPU (pinned) batch
        
        with nvtx.range(f"h2d1-{num_steps}"):
            with torch.cuda.stream(copy_stream):
                next_X = host_X.to(device, non_blocking=True)
                next_Y = host_Y.to(device, non_blocking=True)


        while True:

            if num_steps == warmup_steps:
                nvtx.range_push("POST-WARMUP")  # when ready to measure

            # Wait for the transfer on copy-stream to finish, then use the tensors on default stream
            #Only the first iteration results in "a block"
            #  (i.e., there is no compute on the gpu to overlap the x-fer!)
            torch.cuda.current_stream().wait_stream(copy_stream)
            X, Y = next_X, next_Y

            # IMPORTANT: keep these allocations alive until the default stream finishes with them
            # Prevents X from being released prematurely 
            X.record_stream(torch.cuda.current_stream())
            Y.record_stream(torch.cuda.current_stream())

            # Kick off transfer of the *following* batch immediately
            with nvtx.range(f"h2d2-{num_steps}"):
                try:
                    host_X, host_Y = next(it)
                    with torch.cuda.stream(copy_stream):
                        next_X = host_X.to(device, non_blocking=True)
                        next_Y = host_Y.to(device, non_blocking=True)
                except StopIteration:
                    next_X = next_Y = None #helps break at the bottom of the loop!


            opt.zero_grad(set_to_none=True)

            with nvtx.range(f"forward-{num_steps}"):
                y_hat = model(X)
            with nvtx.range(f"loss-{num_steps}"):
                computed_loss = loss.getCrossEntropyLossFromClass(Y, y_hat)
                        
            #Set learning rate based on schedule!
            with nvtx.range(f"lr-{num_steps}"):
                t = num_steps
                for group in opt.param_groups: #there are a set of param groups
                    curr_lr = optimizer.getCurrentLearningRateBasedOnSchedule (t, alpha_max, alpha_min, tw,tc)
                    group['lr'] = curr_lr

            with nvtx.range(f"backward-{num_steps}"):
                computed_loss.backward()
            with nvtx.range(f"optimizer_step-{num_steps}"):
                opt.step()

            tokens_handled += Y.numel()      
            num_steps += 1

            if next_X is None or num_steps >= max_steps:
                break

        nvtx.range_pop()