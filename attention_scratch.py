import torch
import torch.nn.functional as F

inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your      (x^1)
    [0.55, 0.87, 0.66], # journey   (x^2)
    [0.57, 0.85, 0.64], # starts    (x^3)
    [0.22, 0.58, 0.33], # with      (x^4)
    [0.77, 0.25, 0.10], # one       (x^5)
    [0.05, 0.80, 0.55]  # step      (x^6)
])

# Simplified Attention
def simple_attention():
    '''
    Inputs is (seq_len, hidden_dim)
    This multiplication takes the dot product of every row with every other row
    (6, 3) * (3, 6) = (6, 6)
    This works intuitively because each element in the matrix is a dot
    product of a row from matrix 1 and a column from matrix 2.
    '''
    attention_scores = inputs @ inputs.T

    # Now all the rows sum up to 1. Note that dim=1 means sum=1 across a row
    attention_weights = F.softmax(attention_scores, dim=1)

    ''' 
    Now each embedding in our sequence has an attention weight
    connecting it to every other embedding in the sequence. 
    To get the context vectors, we need to multiply the scalar attention weight
    of each of the embeddings in the sequence by that embedding vector, and add them up.
    This gives us the weighted sum of all the embedding vectors for one particular embedding vector,
    and the result is the context vector for that item in the sequence.
    To calculate all the context vectors at once, we use matrix multiplication. 
    
    If you're ever lost trying to figure out why a certain matrix multiplication
    yields a certain result, recall the three main views of matrix multiplication:
    the row picture and the column picture and dot product picture. 
    See more here: https://ghenshaw-work.medium.com/3-ways-to-understand-matrix-multiplication-fe8a007d7b26
    And here: https://medium.com/geekculture/4-pictures-of-matrix-multiplication-dbe30cb961a9
    
    Recall that if we have A @ B = C, then the rows of C are linear combinations (sums) of the 
    rows of B, weighted by the corresponding rows of A. So the first row of our context matrix C
    will be the context vector for the first item in our sequence. That means it should be a linear
    combination of all the rows of B, the input vector. The weights for this combining are 
    the entries of the first row of A, our attention weight matrix. 
    
    (6, 6) * (seq_len, embed_dim)
    '''
    context_vectors = attention_weights @ inputs
    return context_vectors

def weighted_attention():

    '''
    Now we want to insert trainable parameters for the attention mechanism. There are three
    places to insert trainable params. We insert the weights into the calculation
    by a linear transformation of the sequence vectors by trainable weight matrices.
    We call these the Query, Key and Value matrices.

    Now, instead of computing a context vector for a particular sequence item,
    we instead compute a context vector for the query vector, which is the seq item projected by the
    Query matrix. We then compute the dot product between the query vector and the key vector.
    The key vector will be another sequence item linearly transformed by the Key matrix.
    This operation will give us the attention score matrix again. Finally, we compute the
    context vector not by the attention scores times the input values, but by the value vectors, which
    are a linear projection of the sequence vectors into the value subspace. One final wrinkle is that
    we normalize the attention scores before we use softmax by dividing them by the root of the
    embedding dimension. Why? This is because if we have large dot products and a large embedding
    dimension, the gradients can vanish due to the softmax function applied to them. So we make
    the attention scores smaller before applying softmax, thereby increasing the softmax output and
    increasing the gradient size.
    '''

    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(6, 6), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(6, 6), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(6, 6), requires_grad=False)

    '''
    To understand why this multiplication works, consider again the three matrix multiplication views,
    this time that each row of the inputs is projected into a new subspace by the weight matrix. 
    '''


    keys = inputs @ W_key
    queries = inputs @ W_query
    values = inputs @ W_value

    attention_scores = queries @ keys.T

    # This attention is often called "scaled dot product attention" because of this root scaling factor
    attention_weights = torch.softmax(attention_scores / 6 ** 0.5, dim=-1)

    context_vectors = attention_weights @ values

    return context_vectors

def masked_attention():
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(6, 6), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(6, 6), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(6, 6), requires_grad=False)

    keys = inputs @ W_key
    queries = inputs @ W_query
    values = inputs @ W_value

    attention_scores = queries @ keys.T
    attention_weights = torch.softmax(attention_scores / 6 ** 0.5, dim=-1)

    # Before we apply the weights to the value matrix, we want to mask out some values.
    mask_simple = torch.tril(torch.ones(6, 6))
    # Multiply mask with attention weights to zero-out the values above the diagonal
    masked_simple = attention_weights * mask_simple
    print("\nMasked Simple (zero-out values above diag):\n", masked_simple)

    # Normalize the attention weights to sum up to 1 again in each row
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    print("\nRow Sums:\n", row_sums)
    masked_simple_norm = masked_simple / row_sums
    print("\nNormalized Masked Attention Weights:\n", masked_simple_norm)

    # Masking with 1's above the diagonal and replacing the 1s with negativt infinity (-inf) values, more efficient
    mask = torch.triu(torch.ones(6, 6), diagonal=1)
    masked = attention_scores.masked_fill(mask.bool(), -torch.inf)
    print("\nAlternate Masking Method with -inf:\n", masked)

    # Normalizing alternate masked attention scores
    attn_weights_different = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
    print("\nAlternate Normalized Masked Attention Weights:\n", attn_weights_different)

    context_vectors = attn_weights_different @ values

    return context_vectors

def multi_head_attention():
    '''
    We now
    '''
    pass