import torch
import embedding
import transformer_block
import rmsnorm
import linear
import utils

class Transformer(torch.nn.Module):
    def __init__ (self, d_model, num_heads, d_ff, vocab_size, num_layers, max_seq_len = 100, theta = 10000, device=None, dtype=None ):
        super().__init__()

        self.emb = embedding.Embedding (vocab_size, d_model, device=device, dtype=dtype)
        self.tfblocks = torch.nn.ModuleList([ transformer_block.TransformerBlock(d_model, 
                                                             num_heads, d_ff, 
                                                             max_seq_len = max_seq_len, 
                                                             theta = theta,
                                                             device=device, dtype=dtype) 
                                                             for _ in range (num_layers) ])
        
        self.final_norm = rmsnorm.RMSNorm(d_model, eps = 1e-5, device = device, dtype = dtype)
        self.final_linear = linear.Linear (d_model, vocab_size, device=device, dtype=dtype)

        self.num_layers = num_layers


    def forward (self, x):
        x = self.emb(x)
        for tf_block in self.tfblocks:
            x = tf_block(x)
        
        x = self.final_norm(x)
        x = self.final_linear(x)

        #x = utils.softmax (x, dim_of_interest=-1) 
        # above is taken care of in cross-entropy efficiently, by just returning logits!

        return x