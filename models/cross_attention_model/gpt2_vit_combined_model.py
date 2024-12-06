import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from constants import EOS_TOKEN_ID
from models.image_encoder import ImageEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


from models.cross_attention_model.gpt2_vit_transformer import GPT2Block

class VisionGPT2Model(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.config = config
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        # Note: the names of these parameter fields are meant to match the names of the
        # state_dict for pre-trained GPT models

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.m_cnfg.vocab_size, self.m_cnfg.embed_dim),  # This is the token embedding
            wpe=nn.Embedding(self.m_cnfg.seq_len, self.m_cnfg.embed_dim),  # This is the positional embedding
            drop=nn.Dropout(self.m_cnfg.emb_dropout),
            h=nn.ModuleList([GPT2Block(self.m_cnfg, self.args) for _ in range(self.m_cnfg.depth)]),
            ln_f=nn.LayerNorm(self.m_cnfg.embed_dim)
        ))

        self.lm_head = nn.Linear(self.m_cnfg.embed_dim, self.m_cnfg.vocab_size, bias=False)

        # Weight Tying
        self.transformer.wte.weight = self.lm_head.weight

        self.image_encoder = ImageEncoder(self.config, self.args)

        self.general_gpt_params = [
            self.transformer.wte,
            self.transformer.wpe,
            self.transformer.ln_f,
            self.lm_head
        ]

        self.gpt_layers = [[
            self.transformer.h[i].ln_1,
            self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,
            self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]

        self.blocks = self.gpt_layers  # More sensibly named reference for unfreezing

        for l in self.gpt_layers:
            self.general_gpt_params.extend(l)

        self.vit_params = [
            self.image_encoder.blocks,
            self.image_encoder.vit_pos_embed,
            self.image_encoder.vit_cls_token,
            self.image_encoder.vit_patch_embed
        ]

    def unfreeze_general_params(self):
        for param in self.general_gpt_params:
            param.requires_grad = True

    def unfreeze_layers(self, epoch):
        self.unfreeze_general_params()

        sched = self.o.train_config.decoder_unfreeze_unified
        blocks_to_unfreeze = sched[epoch]

        pretty_blocks = [x + 1 for x in blocks_to_unfreeze]
        if len(pretty_blocks) > 0:
            print(f'\nUnfreezing GPT layers: {pretty_blocks}')

            for block_num in blocks_to_unfreeze:
                block = self.blocks[block_num]
                for layer in block:
                    for p in layer.parameters():
                        p.requires_grad = True

    def vit_pos_embed(self, x):
        pos_embed = self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)

    def pretrained_layers_trainable(self, trainable=False):
        all_params = []
        all_params.extend(self.general_gpt_params)
        all_params.extend(self.vit_params)

        for layer in all_params:
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = trainable
            else:
                layer.requires_grad = trainable

    def unfreeze_gpt_layers(self):
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


    def forward(self, image, token_ids, labels=None):

        # The patch embedding flattens the 2D patches of the image into 1D vectors

        # Batch Size x 3 RGB x 224 x 224
        image_embeddings = self.patch_embed(image) # The patch embedding flattens the 2D patches of the image into 1D vectors

        # Batch Size x 197 x 768 (each 16 by 16 patch is flattened to vector of 196 + 1
        image_embeddings = self.vit_pos_embed(image_embeddings)

        # Batch Size x max sequence length in batch
        token_embeddings = self.transformer.wte(token_ids)  # batch x seq_len

        # 1D Tensor of max sequence length in batch
        positions = torch.arange(0, token_embeddings.size(1)).to(token_embeddings.device)

        # Max sequence length x 768
        positional_embeddings = self.transformer.wpe(positions)

        # Batch Size x max sequence length in batch x 768
        text_input_embeddings = self.transformer.drop(token_embeddings + positional_embeddings)

        # Batch Size x 197 x 768
        vit_hidden_state = image_embeddings

        for i in range(self.config.depth):
            vit_hidden_state = self.vision_blocks[i](vit_hidden_state)

        # Batch Size x max sequence length in batch x 768
        gpt_hidden_state = text_input_embeddings
        for i in range(self.config.depth):

            # Note that gpt and vit both have hidden state embedding dimension of 768
            gpt_hidden_state = self.transformer.h[i](gpt_hidden_state, vit_hidden_state)

        # Batch Size x max sequence length in batch x 768
        gpt_hidden_state = self.transformer.ln_f(gpt_hidden_state)

        if labels is not None:
            # Batch Size x max sequence length in batch x Vocab Size (50257)
            lm_logits = self.lm_head(gpt_hidden_state)
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            # 1D tensor with single value
            return loss

        # Batch Size x Sequence of generated tokens x 768
        # Batch Size is 32 for validation set run and 1 for single image caption generation
        # Note that this gpt_hidden_state[:, [-1], :] preserves the sequence dimension
        last_embedding_in_sequence_of_hidden_states = gpt_hidden_state[:, [-1], :]

        # Batch Size x 1 x Vocab Size
        lm_logits = self.lm_head(last_embedding_in_sequence_of_hidden_states)

        return lm_logits

    def generate(self, image, token_ids_generated_so_far, max_tokens=50, temperature=1.0, sampling_method='argmax'):
        for _ in range(max_tokens):
            # Initially during generation, the tokens tensor only contains token 50256, the start token

            # Batch Size x 1 x Vocab Size
            logits = self.forward(image, token_ids_generated_so_far)

            # Note that this slice operation will remove the sequence length dimension.
            scaled_logits = logits[:, -1, :] / temperature

            # Note that only selecting the last element of the sequence dimension eliminates that dimension
            # So we go from a shape of [batch, sequence, vocab] to [batch, vocab].
            # This 2D tensor is what softmax is expecting.
            probs = F.softmax(scaled_logits, dim=-1)

            if sampling_method == 'argmax':
                next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                try:
                    next_token_id = torch.multinomial(probs, num_samples=1)
                except Exception as e:
                    print(e)
                    next_token_id = torch.tensor([[EOS_TOKEN_ID]]).to(token_ids_generated_so_far.device)

            # Append newly generated token to current token sequence
            token_ids_generated_so_far = torch.cat([token_ids_generated_so_far, next_token_id], dim=1)

            if next_token_id.item() == EOS_TOKEN_ID:
                break

        return token_ids_generated_so_far.cpu().flatten()


    @staticmethod
    def from_pretrained(config, args):

        model = VisionGPT2Model(config, args)
        sd = model.state_dict()

        # These layers are of course not present in the generic GPT2 model
        ignore_matches = ['blocks.', 'cross_attn.', 'ln_3', 'cls_token', 'pos_embed', 'patch_embed.', '.attn.mask']

        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_state_dict = gpt2_small.state_dict()

        gpt2_sd_keys = gpt2_state_dict.keys()
        gpt2_sd_keys = [k for k in gpt2_sd_keys if not k.endswith('.attn.masked_bias')]
        gpt2_sd_keys = [k for k in gpt2_sd_keys if not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in gpt2_sd_keys:
            if any(match in k for match in ignore_matches):
                continue
            if any(k.endswith(w) for w in transposed):
                assert gpt2_state_dict[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(gpt2_state_dict[k].t())
            else:
                assert gpt2_state_dict[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(gpt2_state_dict[k])

        model.load_state_dict(sd)

        return model