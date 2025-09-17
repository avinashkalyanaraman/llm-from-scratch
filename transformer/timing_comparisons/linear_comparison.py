import torch
import timeit

class myLinear (torch.nn.Module):
    def __init__ (self, in_dim, out_dim, device=None):
        super().__init__()

        self.W = torch.randn (in_dim, out_dim, device=device)
    
    def forward(self, x):
        x_prime = x.view(-1, x.shape[-1])
        out = x_prime @ self.W
        out = out.view( *(x.shape[:-1]),out.shape[-1])
        return out
    
    def forward2 (self, x):
        out = x @ self.W
        return out

    def forward3 (self, x):
        out = torch.matmul (x, self.W)
        return out

if __name__ ==  '__main__':

    raws = []
    natives = []
    torch_matmuls = []

    warmups = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    for _ in range (20):
        torch.manual_seed (42)
        input = torch.randn (32,64,384, device=device)
        model = myLinear (384, 1024, device=device)


        start = timeit.default_timer()
        out = model (input)
        end = timeit.default_timer()
        elapsed = end - start
        raws.append (elapsed)
        #print (f"elapsed = {elapsed}")



        start = timeit.default_timer()
        out2 = model.forward2 (input)
        end = timeit.default_timer()
        elapsed = end - start
        natives.append(elapsed)
        #print (f"elapsed = {elapsed}")

        start = timeit.default_timer()
        out3 = model.forward3 (input)
        end = timeit.default_timer()
        elapsed = end - start
        torch_matmuls.append(elapsed)
        #print (f"elapsed = {elapsed}")

        assert torch.equal (out, out2)
        assert torch.equal (out2, out3)


    import statistics
    print (f"mean of raw = {statistics.mean(raws[warmups:])*1000.}ms")
    print (f"mean of natives = {statistics.mean(natives[warmups:])*1000.}ms")
    print (f"mean of torch.matmul = {statistics.mean(torch_matmuls[warmups:])*1000.}ms")