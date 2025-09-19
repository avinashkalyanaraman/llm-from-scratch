import torch
import numpy as np

'''
#The inefficient way! defeats mmap purpose because self.x/y are loading entire content to memory!
class myDatasetInefficient (torch.utils.data.Dataset):
    def __init__ (self, filepath, batch_size=1, context_len=1024, device=None):

        self.contents = np.load (filepath, mmap_mode='r')
        num_elements = len(self.contents)

        self.x = []
        self.y = []
        
        for ii in range (num_elements - context_len):
            self.x.append ( self.contents[ii:ii+context_len] )
            self.y.append ( self.contents [ii+1:ii+context_len+1])

        self.x = torch.tensor (self.x)
        self.y = torch.tensor (self.y)

    def __len__(self):
        return self.x.shape[0]


    def __getitem__(self, idx):
        return (self.x[idx,:], self.y[idx,:])
    

#The more efficient way
class myDatasetInefficient2 (torch.utils.data.Dataset):
    def __init__ (self, filepath, context_len=1024):

        self.contents = np.load (filepath, mmap_mode='r')
        self.context_len = context_len
        self.num_sequences = max(len(self.contents) - context_len, 0) #handles case when len(contents) is < context_len
        
        
    def __len__(self):
        return  self.num_sequences


    def __getitem__(self, idx):
        x_val = self.contents [idx :idx+self.context_len]
        y_val = self.contents [idx+1 : idx+1+self.context_len]

        return torch.from_numpy(x_val.copy()), torch.from_numpy(y_val.copy())  #DataLoader expects torch.Tensor. This helps!
    

'''


#The even more efficient way. It prevents each __getitem__ to hit the disk.
#The more efficient way
class myDataset (torch.utils.data.Dataset):
    def __init__ (self, filepath, context_len=1024, cache_size = 10_000_000):

        self.contents = np.load (filepath, mmap_mode='r+')
        self.context_len = context_len
        self.num_sequences = max(len(self.contents) - context_len, 0) #handles case when len(contents) is < context_len

        self.cache = None
        self.cache_size = cache_size #in a cache
        self.cachestart = -1
        
        
    def __len__(self):
        return  self.num_sequences


    def __getitem__(self, idx):

        #Check if it is in the cache!
        cache_start = (idx // self.cache_size) * self.cache_size

        #Load the cache
        if cache_start != self.cachestart:
            self.cache = self.contents [cache_start:cache_start+self.cache_size+self.context_len ]
            self.cachestart = cache_start
        
        #It is either loaded into the cache now or it is a cache hit. Lets relatively read into this
        rel_start = idx % self.cache_size
        x_val = self.cache [rel_start : rel_start + self.context_len]
        y_val = self.cache [rel_start + 1 : rel_start + 1 + self.context_len]


        #return torch.from_numpy(x_val.copy()), torch.from_numpy(y_val.copy())  #DataLoader expects torch.Tensor. This helps!
        return torch.from_numpy(x_val), torch.from_numpy(y_val)



        
        '''
        #The above .copy() of the initial implmn is because 
        np.load(..., mmap_mode='r') creates a read-only memory map.
        torch.from_numpy(x_val) creates a tensor that shares memory with the NumPy array.
        PyTorch requires writable memory for tensors because operations like .backward() or in-place updates might try to modify the data.
        Since the underlying array is non-writable, PyTorch emits a warning.

        If we aren't modifying the tensor in the code; we can load mmap as 'r+' , and simply do a torch.from_numpy(x_val). 
        This saves us some memory!
        '''



#Mock code below to write a numpy array to disk, and read it in mmap way!
def mockWriteToDisk (fname):
    import tiktoken, os

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    model = "gpt-4"
    encoding = tiktoken.encoding_for_model(model)

    text = "Hello, how are you doing today?"

    # Encode into tokens (ints)
    tokens = encoding.encode(text)
    print(tokens)  # e.g. [9906, 11, 703, 389, ...]

    # Decode back into text
    decoded = encoding.decode(tokens)
    print(decoded)  # "Hello, how are you doing today?"

    np.save (fname, np.array(tokens))  #can also just pass fname, tokens

def mockReadFromDisk (fname):
    x = np.load(fname, mmap_mode='r')
    print (len(x))

if __name__ == '__main__':
    fname = "temp/temp"
    mockWriteToDisk (fname)
    mockReadFromDisk(fname + ".npy")