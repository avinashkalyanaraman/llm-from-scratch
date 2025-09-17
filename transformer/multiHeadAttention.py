import torch
import linear
import utils
import rope


class MHA (torch.nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len = 100, theta=10000, device=None, dtype=None):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model//num_heads #// for int!

        assert d_model % num_heads == 0 , "num_heads must divide d_model for this implementation!"

        self.Wq = linear.Linear (d_model, d_model, device=device, dtype=dtype) #d_model = dk*num_heads
        self.Wk = linear.Linear (d_model, d_model, device=device, dtype=dtype) #d_model = dk*num_heads
        self.Wv = linear.Linear (d_model, d_model, device=device, dtype=dtype) #d_model = dk*num_heads

        #returns to original dmodel space, if dk*num_heads != d_model
        self.Wo = linear.Linear (d_model, d_model, device=device, dtype=dtype)    

        #Rope (rope is done per head)
        self.rope = rope.ROPE2(theta=theta, dim=self.head_dim, max_seq_len=max_seq_len, device=device) 

        self.d_model = d_model  
        

    
    def forward_slow (self, x, token_positions = None, isRope=True): #x is [batchsize, seqlen, dmodel]
        num_heads = self.num_heads

        seq_len = x.shape[-2]

        mask =torch.tril (torch.ones (seq_len,seq_len, dtype=torch.bool) )
        mask = mask.to(x.device) #move the tensor to appropriate device!


        #Create key, query, value tensors!
        q = self.Wq (x)
        k = self.Wk (x)
        v = self.Wv (x)


        #Let us add token_positions if not present:
        if token_positions is None:
            token_positions = torch.arange (seq_len, device = x.device) #seq_len

        head_outs = []

        #Lets go through one head at a time 
        for head_idx in range(num_heads):
            
            #Inject sections of the q,k and v that is for the given head!
            qi = q[...,head_idx*self.head_dim:(head_idx+1)*self.head_dim]
            ki = k[...,head_idx*self.head_dim:(head_idx+1)*self.head_dim]
            vi = v[...,head_idx*self.head_dim:(head_idx+1)*self.head_dim]

            #Rope them to incl. positional information! No need to do it for value!
            if isRope:
                qi = self.rope (qi, token_positions)
                ki = self.rope (ki, token_positions)

            out = utils.attention (qi, ki, vi, mask) #[batchsize, seqlen, dv] where dv = dmodel//num_heads
            head_outs.append (out)

        #Concatenate them!
        agg_out = torch.cat (head_outs, dim=-1) #[batchsize, seqlen, dv*num_heads] = [batchsize, seqlen, dmodel]

        #Final projection! [ We are already in dmodel space here but this is a generality case]!
        agg_out = self.Wo(agg_out)

        return agg_out
    
    def forward (self, x, token_positions = None, isRope=True): #x is [batchsize, seqlen, dmodel]
        num_heads = self.num_heads
        head_dim = self.head_dim

        seq_len = x.shape[-2]

        mask =torch.tril (torch.ones (seq_len,seq_len, dtype=torch.bool) )
        mask = mask.to(x.device) #move the tensor to appropriate device!


        #Create key, query, value tensors!
        q = self.Wq (x) #[batchsize, seqlen, dmodel]
        k = self.Wk (x) #[batchsize, seqlen, dmodel]
        v = self.Wv (x) #[batchsize, seqlen, dmodel]


        #Let us add token_positions if not present:
        if token_positions is None:
            token_positions = torch.arange (seq_len, device = x.device) #seq_len


        q = q.reshape ( *(x.shape[:-1]), num_heads, head_dim) #[batchsize, seqlen, num_heads, head_dim]
        k = k.reshape ( *(x.shape[:-1]), num_heads, head_dim) #[batchsize, seqlen, num_heads, head_dim]
        v = v.reshape ( *(x.shape[:-1]), num_heads, head_dim) #[batchsize, seqlen, num_heads, head_dim]

        q = q.transpose (1,2) #[batchsize, num_heads, seqlen, head_dim]
        k = k.transpose (1,2) #[batchsize, num_heads, seqlen, head_dim]
        v = v.transpose (1,2) #[batchsize, num_heads, seqlen, head_dim]

        if isRope: #No need to rope v
            q = self.rope (q, token_positions)
            k = self.rope (k, token_positions)


        out = utils.attention (q, k, v, mask) #[batchsize, num_heads, seqlen, head_dim]

        #Invert the operations at the start
        out = out.transpose (1,2)  #[batchsize, seqlen, num_heads, head_dim]
        out = out.reshape (*(out.shape[0:-2]), num_heads*head_dim) #[batchsize, seqlen, dmodel]

        #Final projection! [ We are already in dmodel space here but this is a generality case]!
        out = self.Wo(out)

        return out


if __name__ == "__main__":
    x = torch.randn ( 2, 16 , 64)
    mha = MHA(d_model=64, num_heads=4)
    out = mha(x)
    print (f"out shape = {out.shape}")