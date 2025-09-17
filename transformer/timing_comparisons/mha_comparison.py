import argparse
import sys
sys.path.append("../../assignment2-systems")
sys.path.append("../")
import cs336Basics.cs336_basics.model as cs336_model
import cs336Basics.cs336_basics.optimizer as _optimizer
import torch
import statistics

from transformer_pipeline import Transformer
import optimizer
import loss

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


def benchmark (func, args=None, device=None):
    start = timeit.default_timer()
    retval = func(args)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize() 
    end = timeit.default_timer()
    runtime = end-start
    return retval, runtime

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
    parser.add_argument("--epochs", type=int, default=1, help="num of epochs")
    parser.add_argument("--warmups", type=int, default=1, help="num of warmup epochs before timing!")
    parser.add_argument("--rope_theta", type=int, default=10000, help="num of epochs")
    parser.add_argument("--tcompile", type=bool, default=False, help="torch compile?")
    parser.add_argument("--dff", type=int, default=-1, help="ffsize")
    parser.add_argument("--vocabsize", type=int, default=50304, help="vocab size of the tokenizer used!")
    parser.add_argument("--type", type=str, default="comparison", help="baseline or comparison model to run?")


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
    rope_theta = args.rope_theta
    d_ff = args.dff
    exec_type = args.type

    #Identify d_ff
    if d_ff == -1:
        d_ff = round((8/3*d_model)/64)*64 #aligning to a multiple of 64 for hardware reasons!
        d_ff = max (64, d_ff) # it has to be atleast 64

    #set the device!
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    #Create a random set of input and output
    torch.manual_seed(42)
    sample_input = torch.randint ( 0, vocab_size, (batchsize, seqlen), device = device)
    sample_output = torch.randint(0, vocab_size, (batchsize, seqlen), device=device)

    #Define model and optimizer
    if exec_type == 'baseline':
        model = cs336_model.BasicsTransformerLM(vocab_size, seqlen, d_model, num_layers, 
                                                heads, d_ff, rope_theta)
        optim = _optimizer.AdamW (model.parameters(), lr = lr, betas=(beta1,beta2))
    elif exec_type == 'comparison':
        model = Transformer (d_model=d_model, num_heads=heads, d_ff=None, 
                         vocab_size=vocab_size, num_layers= num_layers,
                          max_seq_len=seqlen, device=device, dtype=torch.float32 )
        optim = optimizer.AdamW ( model.parameters(), lr = lr, betas = (beta1,beta2), eps=1e-8, weight_decay = 1e-2)

    model.to(device)


    #Torch-compile:
    if isTorchCompile:
        if device.type == "mps":
        # Use aot_eager for MPS
            model = torch.compile(model, backend="aot_eager")
        elif device.type == "cpu":
            # Skip torch.compile on M1 CPU to avoid clang/OpenMP errors
            print(f"Skipping torch.compile on CPU to avoid compilation errors")
        else:
            # For CUDA or other devices (if available)
            model = torch.compile(model)

    fw_runtimes = []
    bp_runtimes = []

    total_num_trainable_params= sum([ele.numel() for ele in model.parameters() if ele.requires_grad])
    print (f"total # trainable params in {exec_type} model = {total_num_trainable_params/1e6}M")

    for epoch in range(epochs):

        print (f"epoch = {epoch}")


        #zero-out gradient
        optim.zero_grad()

        #Run forward pass
        y_hat, fw_runtime = benchmark (model, args = sample_input, device=device)
        #print (f"{y_hat.shape}")

        loss = getCrossEntropyLossFromClass (sample_output, y_hat)

        #Run backprop!
        _, bp_runtime = benchmark (loss.backward, args = None, device=device)

        #print (f"loss after epoch {epoch} = {loss.item()}")

        if epoch >= warmups:
            fw_runtimes.append (fw_runtime)
            bp_runtimes.append (bp_runtime)
            print (f"Fwd time = {fw_runtime}")
            print (f"Backprop time = {bp_runtime}")


        #Update gradient!
        optim.step()


    print (f"Mean fw pass after warmup of {warmups} steps= {statistics.mean(fw_runtimes)} s")
    print (f"stdev fw pass after warmup of {warmups} steps= {statistics.stdev(fw_runtimes)} s")
    print (f"Mean bw pass after warmup of {warmups} steps= {statistics.mean(bp_runtimes)} s")
    print (f"stdev bw pass after warmup of {warmups} steps= {statistics.stdev(bp_runtimes)} s")