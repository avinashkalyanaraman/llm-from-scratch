import torch
import torch.nn as nn


class ROPE2(nn.Module):
    def __init__(self, theta: float, dim: int, max_seq_len: int, device=None):
        super().__init__()
        assert dim % 2 == 0, "ROPE requires even dim"

        half_dim = dim // 2
        freq_seq = torch.arange(0, half_dim, device=device) / half_dim
        inv_freq = theta ** -freq_seq  # (half_dim,)

        pos = torch.arange(max_seq_len, device=device)  # (max_seq_len,)
        freqs = torch.outer(pos, inv_freq)              # (max_seq_len, half_dim)

        emb = freqs.repeat_interleave(2, dim=-1)  # (max_seq_len, dim)

        cos_matrix = torch.cos(emb)
        sin_matrix = torch.sin(emb)

        self.register_buffer("cos_matrix", cos_matrix)  # (max_seq_len, dim)
        self.register_buffer("sin_matrix", sin_matrix)  # (max_seq_len, dim)

    
    def rotate_half (self, x): #x : batchsize, seqlen, dim
        x1 =  x[...,::2] #batchsize, seqlen, dim/2
        x2 =  x[...,1::2] #batchsize, seqlen, dim/2

        x2 = -x2 #since that is what is used for the first term of rotation "-sin theta"
        stacked_x = torch.stack ( (x2, x1), dim = -1) #batchsize, seqlen, dim/2, 2

        #return stacked_x.reshape_as (x)
        x = stacked_x.view (x.shape) #can also be returned this way!
        return x

    def forward(self, x, token_positions=None):
        #x is typically either key/query typically of shape : [batch_size, seq_len, dim]
        #token_positions when given is of shape : [batch_size, seq_len]

        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device) # (seq_len)

        cos = self.cos_matrix[token_positions] #(seq_len, dim)
        sin = self.sin_matrix[token_positions] #(seq_len, dim)

        return x * cos + self.rotate_half(x) * sin


class ROPE (torch.nn.Module):
    def __init__(self, theta, dim, max_seq_len, device=None ):
        super().__init__()

        idx = torch.arange(dim, device=device).unsqueeze(0)//2
        pos = torch.arange(max_seq_len, device=device).unsqueeze(1)

        exponents = pos/(theta ** (2*idx/dim))
        cos_matrix, sin_matrix = torch.cos (exponents), torch.sin (exponents)
    
        self.register_buffer("cos_matrix", cos_matrix)
        self.register_buffer("sin_matrix", sin_matrix)
    

    def rotate_half (self, x): #x : batchsize, seqlen, dim
        x1 =  x[...,::2] #batchsize, seqlen, dim/2
        x2 =  x[...,1::2] #batchsize, seqlen, dim/2

        x2 = -x2 #since that is what is used for the first term of rotation "-sin theta"
        
        stacked_x = torch.stack ( (x2, x1), dim = -1) #batchsize, seqlen, dim/2, 2

        #return stacked_x.reshape_as (x)
        x = stacked_x.view (x.shape) #can also be returned this way!
        return x



    def forward(self, x, token_positions = None): 
        #x is typically either key/query typically of shape : [batch_size, seq_len, dim]
        #token_positions when given is of shape : [batch_size, seq_len]
    
        seq_len = x.shape[-2]

        if token_positions is None:
            token_positions = torch.arange (0, seq_len, device=x.device)

        
        #reshaping it to a tensor [batch_size, seqlen, dim]
        #this permits leveraging broadcasting
        #reshaped_x = x.view (-1, x.shape[-2], x.shape[-1])  #batchsize, seqlen, dim
        #reshaped_token_positions = token_positions.view (-1, token_positions.shape[-1]) #batchsize, token_pos

        reshaped_x = x
        reshaped_token_positions = token_positions

        cos_matrix = self.cos_matrix [reshaped_token_positions] #[batchsize, seqlen, dim]
        sin_matrix = self.sin_matrix [reshaped_token_positions] #[batchsize, seqlen, dim]

        #cos_matrix = self.cos_matrix[0:seq_len].unsqueeze(0) #[1,seqlen, dim]
        #sin_matrix = self.sin_matrix[0:seq_len].unsqueeze(0) #[1,seqlen, dim]
        
        reshaped_x = reshaped_x * cos_matrix - self.rotate_half (reshaped_x) * sin_matrix

        x = reshaped_x.view (*(x.shape)) #changing back to orig dim
        return x

        #return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"



    
def test_rope():
    rope = ROPE2(theta=10000, dim=8, max_seq_len=20, device='cpu')

    x = torch.randn(2, 10, 8)
    out = rope(x)
    assert out.shape == x.shape, "Shape mismatch"

    # Norm preservation test
    x_norm = x.norm(dim=-1)
    out_norm = out.norm(dim=-1)
    assert torch.allclose(x_norm, out_norm, atol=1e-5), "Norm not preserved"
    print ("first call successful!")

    # Manual simple test
    x_simple = torch.zeros(1, 5, 8)
    x_simple[..., 0::2] = 1
    x_simple[..., 1::2] = 0

    out_simple = rope(x_simple)
    #print("Input:", x_simple)
    #print("Output:", out_simple)
    x_norm = x_simple.norm(dim=-1)
    out_norm = out_simple.norm(dim=-1)
    assert torch.allclose(x_norm, out_norm, atol=1e-5), "Norm not preserved"
    print ("second call successful!")


if __name__ == '__main__':
    test_rope()