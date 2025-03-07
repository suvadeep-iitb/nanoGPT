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
    def __init__(self, n_embd, block_size, attn_span, input_type, bias=True):
        """
        Args:
            n_embd: embedding dimension
            block_size: length of the sequence block
            attn_span: maximum attention span for each query
            input_type: can be one of ['x', 'qkv']
            ln_bias: Whether to use bias in ln layer
        """
        super(LocalAttentionParallel, self).__init__()

        self.n_embd = n_embd
        self.block_size = block_size
        self.attn_span = attn_span
        self.input_type = input_type
        self.bias = bias
        # Create the causal mask
        mask = self.get_mask()
        self.register_buffer('mask', mask, persistent=False)
        # Create the network for query, key, value
        if input_type == 'x':
            self.qkv_net = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.ln = LayerNorm(self.n_embd, bias=self.bias)


    def get_mask(self):
        mask = torch.tril(torch.ones([1, self.block_size, self.block_size]), 0) \
                * (1.0 - torch.tril(torch.ones([1, self.block_size, self.block_size]), -self.attn_span))
        return mask


    def forward(self, inp):
        if self.input_type == 'x':
            assert len(inp) == 1
            x = inp[0]
            q, k, v = self.qkv_net(x).split(self.n_embd, dim=-1)
        elif self.input_type == 'qkv':
            assert len(inp) == 3
            q, k, v = inp
        else:
            assert False
        scaling_factor = np.sqrt(self.n_embd * self.attn_span)
        attn_scores = torch.einsum('bid,bjd->bij', (q, k)) / scaling_factor
        attn_scores = attn_scores * self.mask
        output = self.ln(torch.einsum('bij,bjd->bid', (attn_scores, v)))
        return output


class LinearAttentionParallel(nn.Module):
    """
    Parallel implementation of LinearAttention
    """
    def __init__(self, n_embd, block_size, input_type, bias=True):
        """
        Args:
            n_embd: embedding dimension
            block_size: length of the sequence block
            input_type: can be one of ['x', 'qkv']
            ln_bias: Whether to use bias in ln layer
        """
        super(LinearAttentionParallel, self).__init__()

        self.n_embd = n_embd
        self.block_size = block_size
        self.input_type = input_type
        self.bias = bias
        # Create the causal mask
        mask = self.get_mask()
        self.register_buffer('mask', mask, persistent=False)
        # Create the network for query, key, value
        if self.input_type == 'x':
            self.qkv_net = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.ln = LayerNorm(self.n_embd, bias=self.bias)


    def get_mask(self):
        mask = torch.tril(torch.ones([1, self.block_size, self.block_size]), 0)
        return mask


    def forward(self, inp):
        if self.input_type == 'x':
            assert len(inp) == 1
            x = inp[0]
            q, k, v = self.qkv_net(x).split(self.n_embd, dim=-1)
        elif self.input_type == 'qkv':
            assert len(inp) == 3
            q, k, v = inp
        else:
            assert False
        scaling_factor = np.sqrt(self.n_embd * self.block_size)
        attn_scores = torch.einsum('bid,bjd->bij', (q, k)) / scaling_factor
        attn_scores = attn_scores * self.mask
        output = self.ln(torch.einsum('bij,bjd->bid', (attn_scores, v)))
        return output

