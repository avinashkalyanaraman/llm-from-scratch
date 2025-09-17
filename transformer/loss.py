import torch

def getCrossEntropyRHS (q):
    
    max_q, _ = torch.max (q, dim=-1, keepdim=True) #[batchsize, seqlen, 1]
    q = q - max_q #[batch, seqlen, vocabsize] - [batch, seqlen, 1] --> broadcasted to [batch, seqlen, vocabsize]

    exp_x = torch.exp (q)
    sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True) #[batch, seqlen, 1]

    rhs = q - torch.log(sum_exp_x)

    return rhs

#p is the GT distribution
#q is the output logits
#p is of size [batchsize, seqlen], q is of size [batch, seqlen, vocabsize]. P is assumed to denote best class!
#A more efficient version of this is below.
def getCrossEntropyLossFromClass2 (p,q):
    rhs = getCrossEntropyRHS(q)

    #Now we need to do the equivalrnt of:
    #for i in Batch:
    # for j in seqlen:
    #   r[i,j] = q[i,j, p[i,j]]

    #To do the above we can gather:
    p = p.to(torch.int64) #For gather int64 type needed!
    rhs = torch.gather(rhs, dim=-1, index=p.unsqueeze(-1)) #[batchlen, seqlen, 1]
    rhs = rhs.squeeze(-1)
    all_loss = -rhs
    return torch.mean(all_loss)

#p is the GT distribution
#q is the output logits
#They're of size [batch, seqlen, vocabsize]. P is assumed to be 1-hot-encoded!
def getCrossEntropyLossFromOneHot (p, q):
    
    rhs = getCrossEntropyRHS(q)
    all_loss = -p * rhs

    return torch.mean(all_loss)


#Faster + more memory-efficient cross entropy calculator!
def getCrossEntropyLossFromClass(p, q):
    """
    p: [B, S] targets
    q: [B, S, V] logits
    """
    p = p.to(torch.long)

    m, _ = q.max(dim=-1)                              # [B, S]
    q_shift = q - m.unsqueeze(-1)                     # [B, S, V]
    sumexp = torch.exp(q_shift).sum(dim=-1)           # [B, S]
    lse = m + torch.log(sumexp)                       # [B, S] 

    #lse is the denominator term! log(\sigma (e^(xj))) is written as log(e^max. \sigma (e^(xj - max)) )
    #this becomes m + log (\sigma (e^(xj - max))) <-- this is actually m + torch.log (sumexp)

    # Target logits: the ones that are actually of interest!
    #Now we need to do the equivalrnt of:
    #for i in Batch:
    # for j in seqlen:
    #   r[i,j] = q[i,j, p[i,j]]
    tgt = q.gather(-1, p.unsqueeze(-1)).squeeze(-1)   # [B, S]

    return (lse - tgt).mean()



def getPerplexityFromOneHot (p, q):
    rhs = getCrossEntropyRHS (q) ##[batch, seqlen, vocabsize]

    all_loss = -p * rhs
    avg_loss_per_seq = torch.mean (torch.sum(all_loss, dim = -1), dim = -1)
    perplexity_per_seq = torch.exp (avg_loss_per_seq)
    return perplexity_per_seq

def getPerplexityFromClass (p, q):
    rhs = getCrossEntropyRHS(q)

    #Now we need to do the equivalrnt of:
    #for i in Batch:
    # for j in seqlen:
    #   r[i,j] = q[i,j, p[i,j]]

    #To do the above we can gather:
    rhs = torch.gather(rhs, dim=-1, index=p.unsqueeze(-1)) #[batchlen, seqlen, 1]
    rhs = rhs.squeeze(-1) #[batchlen, seqlen]
    all_loss = -rhs

    avg_loss_per_seq = torch.mean (all_loss, dim=-1)  #[batchlen]
    perplexity_per_seq = torch.exp (avg_loss_per_seq)    #[batchlen]
    return perplexity_per_seq 