import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn

from models.gpt2_attention import GPT2Attention
from models.gpt2_cross_attention import GPT2CrossAttention
from models.gpt2_mlp import GPT2MLP


class GPT2Block(nn.Module):
    # This is the main block for the model
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = GPT2CrossAttention(config)

    def forward(self, x, enc_out):
        shortcut = x
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), enc_out, enc_out)
        x = x + self.mlp(self.ln_3(x))
        x = x + shortcut
        return x