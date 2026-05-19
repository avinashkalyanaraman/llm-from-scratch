import argparse
import sys
sys.path.append("../")
import torch
import statistics

from transformer_pipeline import Transformer
import optimizer, filehandler
import loss

from torch.utils.data import DataLoader


import timeit

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

def slow_gradientClipping (params, max_l2norm, eps = 1e-6):

    #Because we are likely passed an iterator
    params = [p for p in params if p.grad is not None]


    l2_norm_sq= sum([(ele.grad**2).sum() for ele in params if ele.grad is not None]) #returns a tensor (1,). 
    import math
    l2_norm = math.sqrt (l2_norm_sq.item()) #Don't use math.sqrt() ; it causes a gpu2cpu transfer
    
    '''
    .item() moves each tensor to CPU as a Python float, which can be slightly slower on GPU.
    '''

    #We want to avoid the "if" which compares torch scalar w/ cpu scalar causing a gpu2cpu x-fer

    if l2_norm > max_l2norm:
        scaling_factor = max_l2norm/(l2_norm + eps)
        #do clipping

        for param in params:
            if param.grad is None:
                continue
            
            param.grad.data.mul_(scaling_factor)   #in-place
    


def gradientClipping (params, max_l2norm, eps = 1e-6):

    #Because we are likely passed an iterator
    params = [p for p in params if p.grad is not None]

    l2_norm_sq= sum([(ele.grad**2).sum() for ele in params if ele.grad is not None]) #returns a tensor (1,). 
    #the list before sum() contains references to tensors on gpu. no d2h x-fer
    l2_norm = torch.sqrt (l2_norm_sq) 
    
    scaling_factor = (max_l2norm/(l2_norm + eps)).clamp(max=1.0) #eliminates the "if" condition

    for param in params:
        if param.grad is None:
            continue
        
        param.grad.data.mul_(scaling_factor) #in-place!



if __name__ == '__main__' :

    #Get a list of hyper-params as CLI
    #Learning rate, beta1, beta2, tokenizer, d_model, context-len, dff, n-heads, p-enc, 
    # Create the parser
    parser = argparse.ArgumentParser(description="Transformer Run")
    parser.add_argument("--lr", default = 1e-3, type=float, help="Learning Rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for adamw")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for adamw")
    parser.add_argument("--d_model", type=int, default=384, help="dmodel for transformer")
    parser.add_argument("--seqlen", type=int, default=1024, help="context length")
    parser.add_argument("--heads", type=int, default=12, help="number of heads")
    parser.add_argument("--batchsize", type=int, default=1, help="batchsize")
    parser.add_argument("--num_layers", type=int, default=1, help="num of layers")
    parser.add_argument("--num_steps", type=int, default=2, help="num of steps [incl. warmups]")
    parser.add_argument("--warmups", type=int, default=1, help="num of warmup steps before timing!")
    parser.add_argument("--rope_theta", type=int, default=10000, help="parameter for rope")
    parser.add_argument("--dff", type=int, default=-1, help="ffsize")
    parser.add_argument("--vocabsize", type=int, default=50304, help="vocab size of the tokenizer used!")
    parser.add_argument("--epochs", type=int, default=1, help="num of epochs")
    
    #GPU
    parser.add_argument("--tcompile", action="store_true", help="Enable torch.compile",)
    parser.add_argument("--tcore", action="store_true", help="Enable Tensor Core-friendly settings",)
    parser.add_argument("--async_xfer", action="store_true", help="Use pinned memory and async DMA transfers to the GPU",)
    parser.add_argument("--remove_cpu_syncs", action="store_true", help="Remove CPU synchronizations from bad patterns like math.sqrt() and GPU-dependent if-blocks",)

    #Disk
    parser.add_argument("--tfile", type=str, default="temp/temp.npy", help="file with tokens to be used as training")
    parser.add_argument("--datacache", action="store_true",  help="whether to use the dataloader mmap cache")
    parser.add_argument("--shuffle", action="store_true",  help="whether to shuffle in the dataloader")

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
    isTensorCore = args.tcore
    isAsyncXfer = args.async_xfer
    isRmCPUSyncs = args.remove_cpu_syncs
    num_steps = args.num_steps
    warmups = args.warmups
    epochs = args.epochs
    vocab_size = args.vocabsize
    rope_theta = args.rope_theta
    d_ff = args.dff
    fname = args.tfile
    datacache = args.datacache
    isShuffle = args.shuffle

    #Identify d_ff
    if d_ff == -1:
        d_ff = round((8/3*d_model)/64)*64 #aligning to a multiple of 64 for hardware reasons!
        d_ff = max (64, d_ff) # it has to be atleast 64

    #set the device!
    if not torch.cuda.is_available():
        raise RuntimeError("This run is for CUDA devices only")
    device = torch.device("cuda")

    #Create a random set of input and output
    torch.manual_seed(42)

    #Created in the CPU because we want to 'simulate' the effect of
    #reading from disk onto the host memory
    #sample_input = torch.randint ( 0, vocab_size, (batchsize, seqlen), device = 'cpu')
    #sample_output = torch.randint(0, vocab_size, (batchsize, seqlen), device='cpu')

    if datacache:
        train_dataset = filehandler.myDataset (fname,seqlen)
    else:
        train_dataset = filehandler.myDatasetInefficient2 (fname, seqlen)

    train_dataloader = DataLoader (train_dataset, batch_size=batchsize, shuffle=isShuffle, 
                                   drop_last=True, pin_memory=isAsyncXfer)


    #Define model and optimizer
    model = Transformer (d_model=d_model, num_heads=heads, d_ff=d_ff, 
                        vocab_size=vocab_size, num_layers= num_layers,
                        theta = rope_theta, max_seq_len=seqlen,
                        device=device, dtype=torch.float32 )
    optim = optimizer.AdamW ( model.parameters(), lr = lr, betas = (beta1,beta2), eps=1e-8, weight_decay = 1e-2)

    model.to(device)

    #TensorCore
    if isTensorCore:
        # Enable TF32 tensor cores for FP32 matmuls/convs
        torch.set_float32_matmul_precision("high")


    #Torch-compile:
    if isTorchCompile:
        model = torch.compile(model)

    #if isAsyncXfer:
    #    sample_input = sample_input.pin_memory()
    #    sample_output = sample_output.pin_memory()
    

    total_num_trainable_params= sum([ele.numel() for ele in model.parameters() if ele.requires_grad])
    print (f"total # trainable params in model = {total_num_trainable_params/1e6}M")
    
    assert warmups > 0 , "Atleast 1 warmup"
    assert warmups < num_steps , "# of warmup steps < # of steps we want to train!"
    
    timed = False #Did we time the run? Or were our args so small that we didn't run enough warmups and the training exited!
    done = False #For the inner loop to signal to the outer loop below that we are done training!

    curr_step_idx = 0
    
    for epoch in range(epochs):
    
        for batchnum, (X,Y) in enumerate (train_dataloader):             


            X = X.to(device, non_blocking=isAsyncXfer)
            Y = Y.to(device, non_blocking=isAsyncXfer)

            #zero-out gradient
            optim.zero_grad()

            #Run forward pass
            y_hat = model(X)
            #print (f"{y_hat.shape}")

            loss = getCrossEntropyLossFromClass (Y, y_hat)

            #Run backprop!
            loss.backward()

            #The two styles of gradient clips to time!
            if isRmCPUSyncs: #This is a CPU-conditional. no stall!
                gradientClipping(list(model.parameters()), 1)
            else:
                slow_gradientClipping(list(model.parameters()), 1)


            #Update gradient!
            optim.step()


            if curr_step_idx == warmups - 1:
                torch.cuda.synchronize()
                timed = True
                start = timeit.default_timer()

            curr_step_idx += 1

            if (curr_step_idx >= num_steps): #No more epochs to run. We have trained enough!
                done = True
                break
        
        if done:
            break


    if timed and curr_step_idx == num_steps: #We have timed as many steps as asked!
        torch.cuda.synchronize()
        end = timeit.default_timer()
        runtime = end-start
        print (f"Total runtime = {runtime:0.2f}s for {num_steps - warmups} steps")
    else:
        print ("Did not have enough steps for timing!")
