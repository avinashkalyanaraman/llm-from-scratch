import torch
import rmsnorm
import multiHeadAttention
import ffn



class TransformerBlock (torch.nn.Module):
    def __init__ (self, d_model, num_heads, d_ff, max_seq_len = 100, theta=10000, device=None, dtype=None):
        super().__init__()
        self.norm1 = rmsnorm.RMSNorm (d_model, eps = 1e-5, device = device, dtype = dtype)
        self.mha = multiHeadAttention.MHA (d_model, num_heads, max_seq_len = max_seq_len, theta=theta, device = device, dtype = dtype)
        self.norm2 = rmsnorm.RMSNorm (d_model, eps = 1e-5, device = device, dtype = dtype)
        self.ffn = ffn.FFN (d_model, d_ff = d_ff, device = device, dtype = dtype)



    def forward (self, x):
        x1 = self.norm1 (x)
        x1 = self.mha (x1)
        x = x + x1 #The residual connection!

        x1 = self.norm2(x)
        x1 = self.ffn (x1)
        x = x + x1 #The residual connection

        return x