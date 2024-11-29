import warnings

import torch

from models.image_encoder import ImageEncoder
from models.unified_attention_model.gpt2_transformer import GPT2Block
import torch.nn.functional as F
from einops import rearrange

warnings.filterwarnings("ignore")
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class GPT(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.config = config
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embed_dim),  # This is the token embedding
            wpe=nn.Embedding(config.seq_len, config.embed_dim),  # This is the positional embedding
            drop=nn.Dropout(config.emb_dropout),
            h=nn.ModuleList([GPT2Block(config, self.args) for _ in range(config.depth)]),
            ln_f=nn.LayerNorm(config.embed_dim)
        ))

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight Tying
        self.transformer.wte.weight = self.lm_head.weight

        self.image_encoder = ImageEncoder(self.config, self.args)

    def pretrained_layers_trainable(self, trainable=False):
        # Note: the names of these variables are meant to match the names of the
        # state_dict for pre-trained GPT models

        layers = [
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

    def create_unified_input(self, token_ids, image):
        '''
        One fortunate thing is that the image sequences are all the same length,
        so we don't need to worry about padding modifications. I'm pre-pending a
        fixed-length image sequence to each of the text input sequences, so
        the addition of the new items won't affect the padding at the end of the sequences,
        and for a given batch the lengths of the training examples will still be uniform.
        '''

        token_embeddings = self.transformer.wte(token_ids)
        positions = torch.arange(0, token_embeddings.size(1)).to(token_embeddings.device)
        positional_embeddings = self.transformer.wpe(positions)
        text_embeddings = self.transformer.drop(token_embeddings + positional_embeddings)

        image_patch_embeddings = self.image_encoder.vit_patch_embed(image)
        image_embeddings = self.image_encoder.pos_embed(image_patch_embeddings)

        # Now we re-apply the image encoder positional encoding to the encoded image embeddings
        # I could have trained new position embeddings but re-use is simpler and more efficient
        # and my assumption is that this choice won't have a significant negative impact on my results.

        pos_image_embeddings = self.image_encoder.vit_pos_embed + image_embeddings

        unified_embeddings = torch.concat((text_embeddings, pos_image_embeddings), dim=1)

        return unified_embeddings

    def forward(self, image, token_ids, labels=None):
        '''

        Basic training loop is as follows. The training data is the image and the text
        caption. The text caption sequence of tokens are the labels/targets, and
        we input the original caption sequence shifted by 1 along with the sequenced image
        into the model. The model attends to the text part of the input using the text and the
        image. The output is a sequence of hidden states that correspond to the next text token
        in the original caption. We use cross entropy loss to compare the output sequence to the label and
        adjust weights with backprop.
        '''
        input_embeddings = self.create_unified_input(token_ids, image)

        hidden_state = input_embeddings
        for i in range(self.config.depth):
            hidden_state = self.transformer.h[i](hidden_state)

        hidden_state = self.transformer.ln_f(hidden_state)

        # If labels is not None, we are in training mode
        if labels is not None:
            # We need to get rid of the image sequences before we compute the logits
            len_token_ids = token_ids.shape[1]
            hidden_state = hidden_state[:, :len_token_ids, :]

            # lm_head layer projects the hidden state dimension 768 to the vocab dimension 50257
            lm_logits = self.lm_head(hidden_state)

            unrolled_labels = rearrange(labels, 'batch seq -> (batch seq)')
            unrolled_logits = rearrange(lm_logits, 'batch seq vocab -> (batch seq) vocab')

            loss = F.cross_entropy(unrolled_logits, unrolled_labels)
            return loss

        # This is inference mode text generation, where we predict the next token from last hidden state of the sequence
        last_hidden_state = rearrange(
            hidden_state,
            'batch seq hidden - > batch 1 hidden', seq=-1
        )

        lm_logits = self.lm_head(last_hidden_state)
        return lm_logits

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

    @staticmethod
    def from_pretrained(config, args):

        model = GPT(config, args)
        sd = model.state_dict()

        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_state_dict = gpt2_small.state_dict()

        gpt2_sd_keys = gpt2_state_dict.keys()
        gpt2_sd_keys = [k for k in gpt2_sd_keys if not k.endswith('.attn.masked_bias')]
        gpt2_sd_keys = [k for k in gpt2_sd_keys if not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in gpt2_sd_keys:

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