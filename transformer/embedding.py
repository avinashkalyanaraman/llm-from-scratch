import torch

class Embedding (torch.nn.Module):
    def __init__ (self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        #truncation parameters
        mu = 0
        sigma = 1
        a = -3
        b = 3

        self.embedding = torch.empty ( (num_embeddings, embedding_dim), dtype=dtype, device = device)
        self.embedding = torch.nn.Parameter (torch.nn.init.trunc_normal_ (self.embedding, mu, sigma,  a, b))

        self.embedding_dim = embedding_dim
        self.vocab_size = num_embeddings

        #print (f"--- embedding model = {self.embedding} --- ")
    

    def forward (self,  token_ids):
        
        #Verbose variant that can be simplified since torch supports tensor indexing!
        '''
        token_ids_reshaped = token_ids.reshape (-1, token_ids.shape[-1])

        batched_embeddings = []

        embeddings_for_a_batch_item = []
        for row in token_ids_reshaped:
            embeddings_for_a_batch_item = torch.vstack([self.embedding[token,:] for token in row])
            batched_embeddings.append (embeddings_for_a_batch_item)
        
        batched_embeddings = torch.vstack(batched_embeddings)
        batched_embeddings = batched_embeddings.reshape ( *(token_ids.shape) , self.embedding_dim)
        '''

        #Leveraging pytorch's tensor based indexing!
        batched_embeddings2 = self.embedding[token_ids]
        #assert torch.equal (batched_embeddings , batched_embeddings2)
        return batched_embeddings2
        
        

if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    a = torch.tensor ( [[1, 3, 5], [2,2,0]], device=device, dtype=torch.long)
    
    embedding_model = Embedding (10, 4, device= device, dtype=torch.float32)
    embeddings_a = embedding_model(a)
    #print (f"embeddings of a = {embeddings_a}")