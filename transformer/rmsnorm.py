import torch

class RMSNorm (torch.nn.Module):
    def __init__(self, d_model, eps = 1e-5, device = None, dtype = None):
        super().__init__()
        self.gi = torch.nn.Parameter (torch.ones ( (1, d_model), device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps
    
    def forward(self, x): #x is of dim [batchsize, seq_len, d_model]

        #To prevent overflow we upcast!
        in_dtype = x.dtype
        x = x.to(torch.float32)

        
        '''
        #Brute force-way!
        reshaped_x = x.reshape (-1, self.d_model) #makes it [batchsize*seqlen, d_model]

        a = torch.square (reshaped_x) #[b*s, d]
        b = (torch.mean (a, dim = -1) + self.eps).unsqueeze(1)  #[b*s,1]
        c = torch.sqrt(b) #[b*s, 1]
        rms = reshaped_x/c #[b*s, d]
        rms = rms*self.gi #[b*s,d] x [1,d] --> broadcasted to [b*s, d]
        x2 = rms.reshape(*(x.shape[:-1]),self.d_model)
        return x2
        '''
        

        #More elegant!
        norm = torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True)+ self.eps) #[b,s,1]
        rmsnorm = x/norm #[b,s,d] / [b,s,1] --> broadcasted to [b,s,d] / [b,s,d]
        rmsnorm = rmsnorm * self.gi #[b,s,d] * [1,d] --> broadcasted to [b,s,d] * [b,s,d]
        
        #assert torch.allclose(rmsnorm, x2, rtol=1e-5, atol=1e-8)

        rmsnorm = rmsnorm.to(in_dtype)
        return rmsnorm


if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    a = torch.randn ((2,4,2), device=device)

    rmsnorm = RMSNorm(a.shape[-1], device=device)
    norm_a = rmsnorm(a)
    print (f"a = {a}")
    print (f"norm_a = {norm_a}")