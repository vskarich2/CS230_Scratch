import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from models.gpt2_transformer import GPT2Block


class VisionGPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        # TODO: Why is pretrained set to False?
        vit = create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0
        )

        self.patch_embed = vit.patch_embed

        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = nn.Dropout(p=0.)

        # Depth here is 12, and these are the ViT blocks in the vision model
        self.vision_blocks = nn.ModuleList([vit.blocks[i] for i in range(config.depth)])

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embed_dim), # This is the token embedding
            wpe=nn.Embedding(config.seq_len, config.embed_dim), # This is the positional embedding
            drop=nn.Dropout(config.emb_dropout),
            h=nn.ModuleList([GPT2Block(config) for _ in range(config.depth)]),
            ln_f=nn.LayerNorm(config.embed_dim)
        ))

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def _pos_embed(self, x):
        pos_embed = self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)

    def pretrained_layers_trainable(self, trainable=False):
        layers = [
            self.cls_token,
            self.patch_embed,
            self.pos_embed,
            self.vision_blocks,
            self.transformer.wte,
            self.transformer.wpe,
            self.transformer.ln_f,
            self.lm_head
        ]

        gpt_layers = [[
            self.transformer.h[i].ln_1,
            self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,
            self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]

        for l in gpt_layers:
            layers.extend(l)

        for layer in layers:
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = trainable
            else:
                layer.requires_grad = trainable

        total_frozen_params = sum([p.numel() for p in self.parameters() if not p.requires_grad])
        print(f'total_frozen_params: {total_frozen_params}')

    def unfreeze_gpt_layers(self, ):
        gpt_layers = [[
            self.transformer.h[i].ln_1, self.transformer.h[i].ln_2,
            self.transformer.h[i].attn, self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]

        flatten = []
        for l in gpt_layers:
            flatten.extend(l)

        for layer in flatten:
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = True
            else:
                layer.requires_grad = True

    @classmethod
    def from_pretrained(self, config):

        model = VisionGPT2Model(config)
        sd = model.state_dict()
        keys = sd.keys()
        ignore_matches = ['blocks.', 'cross_attn.', 'ln_3', 'cls_token', 'pos_embed', 'patch_embed.', '.attn.mask']
        vit_keys = [key for key in keys if any(match in key for match in ignore_matches)]
        gpt_keys = [key for key in keys if key not in vit_keys]

        # TODO: Try a larger GPT2 model here
        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = gpt2_small.state_dict()
        hf_keys = sd_hf.keys()
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.masked_bias')]
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in hf_keys:
            if any(match in k for match in ignore_matches):
                continue
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        model.load_state_dict(sd)

        return model

    def forward(self, image, input_ids, labels=None):

        image = self.patch_embed(image)
        image = self._pos_embed(image)

        token_embeddings = self.transformer.wte(input_ids)  # batch x seq_len
        pos_embs = torch.arange(0, input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        input_ids = self.transformer.drop(token_embeddings + positional_embeddings)

        # This indicates that we make the forward pass simultaneously through both models
        # I don't think this is correct. The image encoding should be done first, completely.

        for i in range(self.config.depth):
            image = self.vision_blocks[i](image) # TODO: Check the dimensions here.
            input_ids = self.transformer.h[i](input_ids, image) # TODO: Check the dimensions here.

        input_ids = self.transformer.ln_f(input_ids)

        if labels is not None:
            lm_logits = self.lm_head(input_ids)
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            return loss

        lm_logits = self.lm_head(input_ids[:, [-1], :])
        return lm_logits

    def generate(self, image, sequence, max_tokens=50, temperature=1.0, deterministic=False, eos_token_id=50256):
        for _ in range(max_tokens):
            out = self(image, sequence)
            out = out[:, -1, :] / temperature
            probs = F.softmax(out, dim=-1)
            if deterministic:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            sequence = torch.cat([sequence, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break

        return sequence.cpu().flatten()