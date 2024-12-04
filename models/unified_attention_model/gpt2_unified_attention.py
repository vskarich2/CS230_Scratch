import warnings

from constants import NUM_IMAGE_TOKENS

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2UnifiedAttention(nn.Module):
    def __init__(self, config):

        # Image tokens can attend to all image tokens but not text tokens.
        # Text tokens can attend to both image tokens and previous text tokens.
        # The sequence order is image_tokens + text_tokens

        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads

        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'

        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.c_attn = nn.Linear(self.embed_dim, self.head_size * self.n_heads * 3, bias=True)
        self.scale = self.head_size ** -0.5

        combined_mask = self.create_unified_attention_mask()

        self.register_buffer('mask', combined_mask.unsqueeze(0).unsqueeze(0))

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def create_unified_attention_mask(self):

        seq_len = self.seq_len  # Total sequence length (image tokens + text tokens)

        # Create the mask for seq_len x seq_len
        full_mask = torch.tril(torch.ones(seq_len, seq_len))

        # Image tokens (0:197) to only attend to image tokens (0:197)
        image_mask = full_mask[:NUM_IMAGE_TOKENS, :NUM_IMAGE_TOKENS]

        # Allow text tokens (197:seq_len) to attend to both image tokens and previous text tokens
        text_mask = full_mask[NUM_IMAGE_TOKENS:, :]

        # Combine image_mask and text_mask into full seq_len x seq_len mask
        combined_mask = torch.zeros(seq_len, seq_len)
        combined_mask[:NUM_IMAGE_TOKENS, :NUM_IMAGE_TOKENS] = image_mask  # Image-to-image
        combined_mask[NUM_IMAGE_TOKENS:, :] = text_mask  # Text-to-image and text-to-text

        return combined_mask
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        q, k, v = self.c_attn(x).chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_size).permute(0, 2, 1, 3)  # batch x n_heads x seq_len x head_dim
        k = k.view(batch_size, seq_len, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_size).permute(0, 2, 1, 3)

        qk_t = (q @ k.transpose(-2, -1)) * self.scale
        qk_t = qk_t.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        qk_t = F.softmax(qk_t, dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ v  # batch x n_heads x t x head_size
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, embed_dim)  # batch x t x embed_dim

        out = self.c_proj(attention)
        out = self.resid_dropout(out)

        return out
