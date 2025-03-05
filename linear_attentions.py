import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class LocalAttentionParallel(nn.Module):
    """
    Parallel implementation of linear attention where attention span for the i-th query is limitated 
    between (-attn_span, -1) for cuasal attention and between (-attn_span, attn_span)
    for non-causal attention)
    """
    def __init__(self, n_embd, n_query_embd, block_size, attn_span, use_query_emb=False, use_key_emb=False, ln_bias=True):
        """
        Args:
            n_embd: embedding dimension
            n_query_embd: dimension of the query embedding
            block_size: length of the sequence block
            attn_span: maximum attention span for each query
            use_query_emb: Whether query embedding is being used
            use_key_emb: Whether key embedding is being used
            ln_bias: Whether to use bias in ln layer
        """
        super(LocalAttentionParallel, self).__init__()

        self.n_embd = n_embd
        self.n_query_embd = n_query_embd
        self.block_size = block_size
        self.attn_span = attn_span
        self.use_query_emb = use_query_emb
        self.use_key_emb = use_key_emb
        self.ln_bias = ln_bias
        # Create the causal mask
        mask = self.get_mask()
        self.register_buffer('mask', mask, persistent=False)
        # Create the network for query, key, value
        #self.qkv_net = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.q_net = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, self.n_embd),
        )
        self.k_net = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, self.n_embd),
        )
        self.v_net = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, self.n_embd),
        )
        if self.use_query_emb:
            self.q_emb_net = nn.Linear(self.n_query_embd, self.n_embd)
        if self.use_key_emb:
            self.k_emb_net = nn.Linear(self.n_query_embd, self.n_embd)
        self.ln = LayerNorm(self.n_embd, bias=self.ln_bias)


    def get_mask(self):
        mask = torch.tril(torch.ones([1, self.block_size, self.block_size]), 0) \
                * (1.0 - torch.tril(torch.ones([1, self.block_size, self.block_size]), -self.attn_span))
        return mask


    def forward(self, x, query_emb=None, key_emb=None):
        B, T, C = x.size()
        if query_emb is not None:
            q = x + self.q_emb_net(query_emb)
        else:
            q = x
        if key_emb is not None:
            k = x + self.k_emb_net(key_emb)
        else:
            k = x
        #q, k, v = self.qkv_net(x).split(self.n_embd, dim=-1)
        q = self.q_net(q)
        k = self.k_net(k)
        v = self.v_net(x)

        scaling_factor = np.sqrt(self.n_embd * self.attn_span)
        attn_scores = torch.einsum('bid,bjd->bij', (q, k)) / scaling_factor
        attn_scores = attn_scores * self.mask
        output = self.ln(torch.einsum('bij,bjd->bid', (attn_scores, v)))
        return output


class LinearAttentionParallel(nn.Module):
    """
    Parallel implementation of LinearAttention
    """
    def __init__(self, n_embd, n_query_embd, block_size, use_query_emb=False, use_key_emb=False, ln_bias=True):
        """
        Args:
            n_embd: embedding dimension
            n_query_embd: dimension of the query embedding
            block_size: length of the sequence block
            use_query_emb: Whether query embedding is being used
            use_key_emb: Whether key embedding is being used
            ln_bias: Whether to use bias in ln layer
        """
        super(LinearAttentionParallel, self).__init__()

        self.n_embd = n_embd
        self.n_query_embd = n_query_embd
        self.block_size = block_size
        self.use_query_emb = use_query_emb
        self.use_key_emb = use_key_emb
        self.ln_bias = ln_bias
        # Create the causal mask
        mask = self.get_mask()
        self.register_buffer('mask', mask, persistent=False)
        # Create the network for query, key, value
        #self.qkv_net = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.q_net = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, self.n_embd),
        )
        self.k_net = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, self.n_embd),
        )
        self.v_net = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, self.n_embd),
        )
        if self.use_query_emb:
            self.q_emb_net = nn.Linear(self.n_query_embd, self.n_embd)
        if self.use_key_emb:
            self.k_emb_net = nn.Linear(self.n_query_embd, self.n_embd)
        self.ln = LayerNorm(self.n_embd, bias=self.ln_bias)


    def get_mask(self):
        mask = torch.tril(torch.ones([1, self.block_size, self.block_size]), 0)
        return mask


    def forward(self, x, query_emb=None, key_emb=None):
        B, T, C = x.size()
        if query_emb is not None:
            q = x + self.q_emb_net(query_emb)
        else:
            q = x
        if key_emb is not None:
            k = x + self.k_emb_net(key_emb)
        else:
            k = x
        #q, k, v = self.qkv_net(x).split(self.n_embd, dim=-1)
        q = self.q_net(q)
        k = self.k_net(k)
        v = self.v_net(x)

        scaling_factor = np.sqrt(self.n_embd * self.block_size)
        attn_scores = torch.einsum('bid,bjd->bij', (q, k)) / scaling_factor
        attn_scores = attn_scores * self.mask
        output = self.ln(torch.einsum('bij,bjd->bid', (attn_scores, v)))
        return output
