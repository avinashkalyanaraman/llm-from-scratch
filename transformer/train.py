import torch
import os
import sys
import argparse
import filehandler
from torch.utils.data import DataLoader


from transformer_pipeline import Transformer
import optimizer
import loss
import checkpoint
import statistics

import timeit

import wandb
import wandb_utils

#How often to checkpoint the model, run-validation and log to wandb!
CHECKPOINT_RATE = 5000
VALIDATION_RATE = 1000
WANDB_LOG_RATE = 50
ISWANDB = True

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
    parser.add_argument("--epochs", type=int, default=1, help="num of epochs")
    parser.add_argument("--vocabsize", type=int, default=50304, help="vocab size of the tokenizer used! Power of 64 helps!")
    parser.add_argument("--tfile", type=str, default="temp/temp.npy", help="file with tokens to be used as training")
    parser.add_argument("--vfile", type=str, default="temp/temp.npy", help="file with tokens to be used for validation")
    parser.add_argument("--time", action="store_true", help="enable timing")
    parser.add_argument("--tokenlimit", type=int, default=40_000_000, help="total # of training tokens to consider" )
    parser.add_argument("--val_tokenlimit", type=int, default=10_000_000, help="total # of validation tokens to consider" )

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
    epochs = args.epochs
    vocab_size = args.vocabsize
    t_fname = args.tfile
    v_fname = args.vfile
    isTime = args.time
    tokenlimit = args.tokenlimit
    val_tokenlimit = args.val_tokenlimit

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if device.type == "cuda":
        # Enable TF32 tensor cores for FP32 matmuls/convs
        torch.set_float32_matmul_precision("high")

    #Wandb Config Settings!
    # Start a new wandb run to track this execution.
    if ISWANDB:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="avinashkaly-self",
            # Set the wandb project where this run will be logged.
            project="llm-from-scratch",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": lr,
                "beta1" : beta1,
                "beta2" : beta2,
                "d_model" : d_model,
                "seqlen" : seqlen,
                "vocab_size" : vocab_size,
                "heads" : heads,
                "batchsize" : batchsize,
                "num_layers" : num_layers,
        #        "architecture": "Transformer",
        #        "dataset": "TinyStoriesV2",
                "epochs": epochs,
            },
        )


    train_dataset = filehandler.myDataset (t_fname,seqlen)
    val_dataset = filehandler.myDataset(v_fname, seqlen)

    print (f"The dataset has  {len(train_dataset)} entires for training")
    print (f"The dataset has  {len(val_dataset)} entires for validation")

    #Checkpoint info
    checkpoint_file = 'checkpoints/step9600.pt'
    final_file = 'checkpoints/final_model.pt'

    # Split the dataset
    train_dataloader = DataLoader (train_dataset, batch_size=batchsize, shuffle=True, drop_last=True) #TODO: set shuffle to True for larger mem
    val_dataloader = DataLoader (val_dataset, batch_size=batchsize*2, shuffle=False, drop_last=True)

    max_val_steps = val_tokenlimit // (val_dataloader.batch_size * seqlen) #denominator is tokens per step


    model = Transformer (d_model=d_model, num_heads=heads, d_ff=None, 
                         vocab_size=vocab_size, num_layers= num_layers,
                          max_seq_len=seqlen, device=device, dtype=torch.float32 )
    model.to(device)

    if device.type == "mps":
        # Use aot_eager for MPS
        model = torch.compile(model, backend="aot_eager")
    elif device.type == "cpu":
        # Skip torch.compile on M1 CPU to avoid clang/OpenMP errors
        print(f"Skipping torch.compile on CPU to avoid compilation errors")
    else:
        # For CUDA or other devices (if available)
        model = torch.compile(model)

    opt = optimizer.AdamW ( model.parameters(), lr = lr, betas = (beta1,beta2), eps=1e-8, weight_decay = 1e-2)

    #For cosine annealing:  f(tokenlimit) as opp to f(train_dataloader)
    tc = tokenlimit/(batchsize*seqlen)  #Suggested best practice: is to reach alpha_min at tokenlimit
    tw = 0.05*tc  #warmup fraction is 5% of tc!
    alpha_max = lr
    alpha_min = 1e-6


    num_steps = 0
    tokens_handled = 0
    start_epoch = 0
    start_iter = 0

    warmup_steps = 5

    if  os.path.exists(checkpoint_file):
        num_steps = checkpoint.load_checkpoint (checkpoint_file, model, opt) 
        start_epoch = num_steps // len(train_dataloader)
        start_iter = num_steps % len(train_dataloader)
        tokens_handled = num_steps * batchsize * seqlen
        print (f"tokens handled = {tokens_handled}")
        print (f"token limit = {tokenlimit}")
        

    step_times = []
    throughputs = []
    run_finished=False

    if tokens_handled >= tokenlimit:
        print ("Training is already done!")
        sys.exit(0)


    total_num_trainable_params= sum([ele.numel() for ele in model.parameters() if ele.requires_grad])
    print (f"total # trainable params in model = {total_num_trainable_params/1e6}M")

    for epoch in range(start_epoch, epochs):
        for batchnum, (X,Y) in enumerate (train_dataloader): 
            
            #Checkpoint restoration! 
            if epoch == start_epoch and batchnum < start_iter: 
                continue

            if isTime:
                start = timeit.default_timer()       

            X = X.to(device)
            Y = Y.to(device) 

            opt.zero_grad(set_to_none=True) #Faster as : sets each parameter’s .grad to None instead of a tensor of zeros

            y_hat = model (X)
            #assert y_hat.shape[:-1] == Y.shape, f"{y_hat.shape} vs {Y.shape}"

            computed_loss = loss.getCrossEntropyLossFromClass(Y, y_hat)
            #print (f"Computed loss {computed_loss.item(): 0.3f} at num_step = {num_steps}")

            #Set learning rate based on schedule!
            t = num_steps

            for group in opt.param_groups: #there are a set of param groups
                curr_lr = optimizer.getCurrentLearningRateBasedOnSchedule (t, alpha_max, alpha_min, tw,tc)
                group['lr'] = curr_lr


            computed_loss.backward() #Computes the gradients
            optimizer.gradientClipping (model.parameters(), 1) #Clip the gradients
            opt.step() #Updates the weights

            tokens_handled += Y.numel()
            num_steps += 1
  
            #Timing
            if isTime:
                end = timeit.default_timer()
                elapsed = end - start
                throughput = Y.numel() / max(elapsed, 1e-9)
                step_times.append (elapsed)
                throughputs.append(throughput)

            
            #Checkpoint
            if num_steps % CHECKPOINT_RATE == 0:
                checkpoint.save_checkpoint(model, opt, num_steps, f'checkpoints/step{num_steps}.pt')
            
            #Run validation loop
            if num_steps % VALIDATION_RATE == 0:

                model.eval()
                total_val_loss = torch.zeros ( (), device = device)
                val_steps = 0

                with torch.no_grad():
                    print (f"handling validation")
                    for v_batchnum, (X_v, Y_v) in enumerate(val_dataloader):
                        X_v = X_v.to(device)
                        Y_v = Y_v.to(device=device)
                        val_y_hat = model (X_v)
                        # computed_val_loss = loss.getCrossEntropyLossFromClass(Y_v, val_y_hat).item() #<-- the item() causes gpu->cpu every batch. Avoid!
                        computed_val_loss = loss.getCrossEntropyLossFromClass(Y_v, val_y_hat)
                        total_val_loss += computed_val_loss
                        if v_batchnum % 100 == 0:
                            print (f"validation batchnum = {v_batchnum}")
                        val_steps += 1
                        if val_steps >= max_val_steps:
                            break
                mean_val_loss = total_val_loss.item()*1./val_steps #TODO: Later profile-&-verify, which is faster. accumulating computed_val_loss as .item() for every batch, or this and suffering kernel launch penalty!
                print (f"Mean val loss = {mean_val_loss}")
                model.train() #Go back to training mode!

                if ISWANDB:
                    run.log( {"val_loss" : mean_val_loss}, step = num_steps )

            if (num_steps % WANDB_LOG_RATE) == 0:
                print (f"Computed loss {computed_loss.item(): 0.3f} at num_step = {num_steps}")

            #Log to Wandb
            if ISWANDB and (num_steps % WANDB_LOG_RATE == 0):
                log_dict = { 'curr_lr' : curr_lr, "training_loss" : computed_loss.item(),
                    "tokens" : tokens_handled}
                if isTime:
                    log_dict['steptime'] = elapsed 
                    log_dict['throughput'] = throughput

                run.log (log_dict, step=num_steps)


            #Have we run enough
            if tokens_handled >= tokenlimit:
                checkpoint.save_checkpoint (model, opt, num_steps, final_file)
                run_finished=True
                break

        if run_finished:
            break

    if isTime and len(step_times) > 0: 
        print (f"mean running time = {statistics.mean(step_times[warmup_steps:]):0.2f}s")
        print (f"mean throughput = {statistics.mean(throughputs[warmup_steps:]):0.2f} tokens/sec")
    
    if ISWANDB:
        run.finish()