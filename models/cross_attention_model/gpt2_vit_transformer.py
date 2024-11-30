import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn

from models.cross_attention_model.gpt2_self_attention import GPT2SelfAttention
from models.cross_attention_model.gpt2_vit_cross_attention import GPT2CrossAttention
from models.common.gpt2_mlp import GPT2MLP


class GPT2Block(nn.Module):
    # This is the main block for the model
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2SelfAttention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = GPT2CrossAttention(config)

    def forward(self, x, enc_out):
        if self.args.shortcut:
            shortcut = x
            x = x + self.attn(self.ln_1(x))
            x = x + self.cross_attn(self.ln_2(x), enc_out, enc_out)
            x = x + self.mlp(self.ln_3(x))
            x = x + shortcut
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.cross_attn(self.ln_2(x), enc_out, enc_out)
            x = x + self.mlp(self.ln_3(x))
        return x