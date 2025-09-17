import torch
import numpy as np


class Linear (torch.nn.Module):
    def __init__ (self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        #self.W = torch.randn( (in_features, out_features)) #This will not make it register as a model parameter to get updated in backprop! --> nn.Parameter()
        #self.W = torch.nn.Parameter ( torch.randn( (in_features, out_features), dtype = dtype) )

        #Initializing in a way that facilitates training!
        mu = 0
        sigma = np.sqrt (2/(in_features + out_features))

        self.W = torch.empty ((in_features, out_features), device=device, dtype=dtype)
        self.W = torch.nn.Parameter (torch.nn.init.trunc_normal_ (self.W, mu, sigma,  -3*sigma, 3*sigma))


    def forward (self, x):
        

        '''
        #Explicit reshape handled
        reshaped_x = x.reshape (-1, x.shape[-1]) #reshape w/ -1 asks pytorch to identify the appropriate dimension
        y = reshaped_x @ self.W
        y2 = y.reshape(*(x.shape[:-1]), y.shape [-1]) #unwraps it as args
        '''

        #y1 = torch.matmul (x, self.W) #deals with >2 dimension w/o explicit reshaping
        #assert (torch.equal (y1,y2))

        y3 = x @ self.W
        #assert torch.equal (y3, y2)

        return y3
        
    

if __name__ == '__main__':
    input_len = 3
    layer_size = 4
    batch_size = 5


    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    torch.manual_seed (42)


    linearModel = Linear (input_len, layer_size, dtype=torch.float16, device=device)
    x = torch.randn ((batch_size,input_len), dtype=torch.float16, device=device)
    y = linearModel (x)
    print (f"x = {x}")
    print (f"y = {y} with shape = {y.shape}")
   