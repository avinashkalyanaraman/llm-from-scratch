import torch
from linear import Linear


#We want to implement FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x),
class FFN (torch.nn.Module):
    def __init__(self, dmodel, d_ff = None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            d_ff = round((8/3*dmodel)/64)*64 #aligning to a multiple of 64 for hardware reasons!
            d_ff = max (64, d_ff) # it has to be atleast 64

        self.linear3 = Linear (dmodel, d_ff, device=device, dtype=dtype)
        self.linear1 = Linear (dmodel, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear (d_ff, dmodel, device=device, dtype=dtype)


    def forward (self, x):
        
        u = self.linear1(x) #W1x
        u = u * torch.sigmoid (u) #the SiLu bit
        v = self.linear3(x) #W3x
        z = u * v #element wise mult

        return self.linear2 (z) #linear to get back to orig dims!

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    a = torch.randn ( (5,256), device=device, dtype=torch.float32)
    ffn = FFN (a.shape[1], device = device, dtype=torch.float32)
    out = ffn(a)
    print (f"out.shape = {out.shape}")

