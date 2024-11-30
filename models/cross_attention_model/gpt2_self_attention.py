import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.c_attn = nn.Linear(self.embed_dim, self.head_size * self.n_heads * 3, bias=True)
        self.scale = self.head_size ** -0.5

        self.register_buffer('mask', torch.tril(torch.ones(1, 1, self.seq_len, self.seq_len)))

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x):
        b, t, c = x.shape

        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)  # batch x n_heads x seq_len x head_dim
        k = k.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)

        qk_t = (q @ k.transpose(-2, -1)) * self.scale
        qk_t = qk_t.masked_fill(self.mask[:, :, :t, :t] == 0, float('-inf'))
        qk_t = F.softmax(qk_t, dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ v
        attention = attention.permute(0, 2, 1, 3).contiguous().view(b, t, c)  # batch x t x embed_dim

        out = self.c_proj(attention)
        out = self.resid_dropout(out)

        return out