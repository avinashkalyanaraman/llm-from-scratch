import tiktoken
import os
import argparse
import torch
import optimizer
import checkpoint
import utils

from transformer_pipeline import Transformer

#lets load the model from the latest checkpoint!

if __name__ == '__main__' :

    #Get a list of hyper-params as CLI
    #Learning rate, beta1, beta2, tokenizer, d_model, context-len, dff, n-heads, p-enc, 
    # Create the parser
    parser = argparse.ArgumentParser(description="Transformer Run")
    parser.add_argument("--d_model", type=int, default=384, help="dmodel for transformer")
    parser.add_argument("--seqlen", type=int, default=128, help="context length")
    parser.add_argument("--heads", type=int, default=12, help="number of heads")
    parser.add_argument("--num_layers", type=int, default=1, help="num of layers")
    parser.add_argument("--vocabsize", type=int, default=50304, help="vocab size of the tokenizer used!")
    parser.add_argument("--max_tokens", type=int, default=2048, help="max output tokens to generate (incl input)")
    parser.add_argument("--temperature", type=float, default=1, help="temperature for generation")
    parser.add_argument("--p", type=float, default=1, help="p in top-p")
    parser.add_argument("--sampling", type=str, default="greedy", help="sampling strategy")


    args = parser.parse_args()

    #the local vars
    d_model = args.d_model
    seqlen = args.seqlen
    heads = args.heads
    num_layers = args.num_layers
    vocab_size = args.vocabsize
    max_tokens = args.max_tokens
    temperature = args.temperature
    p = args.p
    sampling = args.sampling


    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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

    #Just reading it, unused. TODO: Handle later
    opt = optimizer.AdamW ( model.parameters())

    #Checkpoint info
    checkpoint_file = 'checkpoints/step10000.pt'
    assert os.path.exists (checkpoint_file) , "Checkpoint file missing!"
    _ = checkpoint.load_checkpoint (checkpoint_file, model, opt)
    print (f"Checkpoint loaded!")

    #get endoftext special token! gpt2 has |endoftext| built-in. lets use that
    #TODO: Now hard-coded fix later, after writing to disk!
    # Encode the built-in end-of-text token
    enc = tiktoken.get_encoding("gpt2")

    endoftext_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    #Tokenize the input!
    input = "tell me a story set in istanbul!"
    input_ids = enc.encode (input)
    input_len = len(input_ids)
    max_tokens = max_tokens - input_len
    #input_ids.extend ([endoftext_token]* max_tokens)
    input_ids = torch.tensor (input_ids, device=device)

    output_string = ""

    input_ids = input_ids.unsqueeze(0)  #For batchsize compatibility!

    for token_num in range(max_tokens):
        logits = model(input_ids) #TODO: Maybe batching is needed!
        #print (f"Logits obtained. And shape of logits = {logits.shape}")

        #Step1: scale w.r.t temperature
        logits = logits/temperature
        logits = logits[:,input_len-1,:] #Only care about the last in the seq; #[batchlen, vocab-size]

        
        #Step2: Sample from that
        #lets do max sampling for now. make sure to only look up to input-len
        if sampling == 'greedy':
            #y_hats = utils.softmax (logits, dim_of_interest=-1) #[batchlen, vocab-size]
            output = torch.argmax (logits, dim = -1) #[batchsize,] we don't need to do above softmax!
            #handle for batchsize = 1 (our case)
            output = output[0].item() #scalar 

        elif sampling == 'nucleus':
            sorted_logits, sorted_indices = torch.sort(logits, dim = -1, descending=True)
            #do softmax on the logits
            y_hats = utils.softmax(sorted_logits, dim_of_interest= -1) #[batchlen, vocabsize]          
            
            #keep only those such that cumsum <= p
            y_hats_cumsum = torch.cumsum (y_hats, dim=-1) #compute cumsum
            #set those <= p to have probability, and rest to 0
            y_hats.masked_fill_ (y_hats_cumsum>p, 0) #[batchlen, vocabsize]
            #renormalize so that they sum to 1 for sampling!
            y_hats = y_hats/torch.sum(y_hats, dim=-1, keepdim=True) #[batchlen, vocabsize]

            #sample from the above normalized tensor to get an index per row
            y_hat_indices = torch.multinomial(y_hats, 1 ) #[batchlen, 1]

            #take only the first one. since we consider batchsize=1
            y_hat_index = y_hat_indices[0,0].item()
            output = sorted_indices[0,y_hat_index] #map back to original vocab index!

        #Step3: if endoftext, break!
        if output == endoftext_token:
            break

        #Step4: else, add it back to input_ids
        #torch.cat should have all dims equal, except dim along which cat is done!
        input_ids = torch.cat ([input_ids, torch.tensor([output], device=device).unsqueeze(0)], dim = -1 )

        #Step5 : Store the output!
        output_text = enc.decode([output])
        output_string = output_string + output_text
        print(f"{output_text}")

        input_len = input_len + 1
    
    print (f"output_string : {output_string}")



    
