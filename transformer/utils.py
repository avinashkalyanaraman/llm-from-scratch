import torch
import torch.nn.functional as F



def softmax (x, dim_of_interest):
    max_x, _ = torch.max(x, dim=dim_of_interest, keepdim=True )
    delta = x - max_x

    exp_delta = torch.exp (delta)
    exp_delta_row_sum = torch.sum (exp_delta, dim=dim_of_interest, keepdim=True)

    return exp_delta / exp_delta_row_sum

def are_broadcastable(*shapes):
    try:
        torch.broadcast_shapes(*shapes)
        return True
    except RuntimeError:
        return False

def attention (q, k, v, mask=None):
    #q, k are of size: [batch,..., seqlen, dk]
    #v is of size : [batch, ..., seqlen, dv]

    dk = k.shape[-1] #dim of key/query

    kT = k.transpose(-2,-1) 

    attn_values = q @ kT/(dk**0.5)

    if mask is not None:
        #assert are_broadcastable ( attn_values.shape, mask.shape), f"Shape mismatch in mask with shapes {mask.shape} and {attn_values.shape}!"
        attn_values.masked_fill_ (~mask, float ('-inf')) #replace wherever condition is met (=~False) with -inf!

    attn_weights = softmax (attn_values, dim_of_interest=-1)
    #print (f"attn_weights shape = {attn_weights.shape}")
    return attn_weights @ v



def test_softmax():
    # Test parameters
    batch_size, seq_len, dim = 4, 5, 6
    dim_of_interest = -1 # For example, softmax along last dimension

    # Generate random input tensor
    x = torch.randn(batch_size, seq_len, dim)

    # Compute softmax using your function
    my_softmax = softmax(x, dim_of_interest)

    # Compute softmax using PyTorch builtin
    torch_softmax = F.softmax(x, dim=dim_of_interest)

    # Test 1: Check output shape matches input shape
    assert my_softmax.shape == x.shape, f"Shape mismatch: {my_softmax.shape} vs {x.shape}"

    # Test 2: Check output sums to 1 along dim_of_interest (within tolerance)
    sums = my_softmax.sum(dim=dim_of_interest)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), "Softmax sums are not 1"

    # Test 3: Compare with torch.softmax for approximate equality
    assert torch.allclose(my_softmax, torch_softmax, atol=1e-6), "Softmax outputs differ from torch's implementation"

    print("All tests passed!")

def test_attention_output_shape():
    batch, seqlen, dk, dv = 2, 3, 4, 5
    q = torch.randn(batch, seqlen, dk)
    k = torch.randn(batch, seqlen, dk)
    v = torch.randn(batch, seqlen, dv)
    out = attention(q, k, v)
    assert out.shape == (batch, seqlen, dv), f"Output shape {out.shape} incorrect"
    print("Attention output shape test passed!")

def test_attention_masking_effect():
    batch, seqlen, dk, dv = 2, 3, 4, 5
    q = torch.randn(batch, seqlen, dk)
    k = torch.randn(batch, seqlen, dk)
    v = torch.randn(batch, seqlen, dv)

    mask = torch.ones(batch, seqlen, seqlen, dtype=torch.bool)
    mask[:, :, 1] = 0  # mask out second key position

    out_masked = attention(q, k, v, mask=mask)
    out_unmasked = attention(q, k, v, mask=torch.ones_like(mask))
    assert not torch.allclose(out_masked, out_unmasked), "Masking did not affect output"
    print("Attention masking effect test passed!")

def test_attention_weights_sum_to_one():
    batch, seqlen, dk, dv = 2, 3, 4, 5
    q = torch.randn(batch, seqlen, dk)
    k = torch.randn(batch, seqlen, dk)
    v = torch.randn(batch, seqlen, dv)

    # Extract attention weights by modifying attention to return weights for test
    def attn_weights_fn(q, k, v, mask=None):
        dk = k.shape[-1]
        kT = k.transpose(-2, -1)
        attn_values = q @ kT / (dk ** 0.5)
        if mask is not None:
            attn_values.masked_fill_(~mask, float('-inf'))
        attn_weights = softmax(attn_values, dim_of_interest=-1)
        return attn_weights

    weights = attn_weights_fn(q, k, v)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), "Attention weights do not sum to 1"
    print("Attention weights sum to one test passed!")


#ChatGPT generated unit tests!
if __name__ == "__main__":
    test_softmax()
    test_attention_output_shape()
    test_attention_masking_effect()
    test_attention_weights_sum_to_one()
