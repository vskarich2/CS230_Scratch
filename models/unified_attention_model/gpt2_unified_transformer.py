import warnings

from models.unified_attention_model.gpt2_unified_attention import GPT2UnifiedAttention

warnings.filterwarnings("ignore")
import torch.nn as nn

from models.common.gpt2_mlp import GPT2MLP

class GPT2UnifiedBlock(nn.Module):
    # This is the main block for the vanilla GPT model
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2UnifiedAttention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

