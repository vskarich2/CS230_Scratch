import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import torch

from constants import EOS_TOKEN_ID, NUM_IMAGE_TOKENS
from models.image_encoder import ImageEncoder
from models.unified_attention_model.gpt2_unified_transformer import GPT2UnifiedBlock
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections.abc import Iterable

import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class UnifiedAttentionModel(nn.Module):
    def __init__(self, o):
        super().__init__()
        self.o = o
        self.m_cnfg = o.model_config
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        # Note: the names of these parameter fields are meant to match the names of the
        # state_dict for pre-trained GPT models

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.m_cnfg.vocab_size, self.m_cnfg.embed_dim),  # This is the token embedding
            wpe=nn.Embedding(self.m_cnfg.seq_len, self.m_cnfg.embed_dim),  # This is the positional embedding
            drop=nn.Dropout(self.m_cnfg.emb_dropout),
            h=nn.ModuleList([GPT2UnifiedBlock(self.m_cnfg, self.o.args) for _ in range(self.m_cnfg.depth)]),
            ln_f=nn.LayerNorm(self.m_cnfg.embed_dim)
        ))

        self.lm_head = nn.Linear(self.m_cnfg.embed_dim, self.m_cnfg.vocab_size, bias=False)

        # Weight Tying
        self.transformer.wte.weight = self.lm_head.weight

        # At this point we only have defined GPT params so this will only print those
        print(f'Total params GPT: {sum([p.numel() for p in self.parameters() if p.requires_grad])}')

        self.image_encoder = ImageEncoder(o)

        '''It is important to surface these here so that they are saved in state_dict
        When we start unfreezing the encoder, it is important not to call forward
        on the encoder, but rather use the params here as these will be the ones'''

        # GPT trainable
        self.gpt_general_params = [
            self.transformer.wte,
            self.transformer.wpe,
            self.transformer.ln_f,
            self.lm_head
        ]

        self.gpt_blocks = [[
            self.transformer.h[i].ln_1,
            self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,
            self.transformer.h[i].mlp
        ] for i in range(self.m_cnfg.depth)]

        # ViT trainable
        self.vit_general_params = [
            self.image_encoder.pos_embed,
            self.image_encoder.cls_token,
            self.image_encoder.patch_embed
        ]

        self.vit_blocks = self.image_encoder.blocks
        print(f'Total combined params: {sum([p.numel() for p in self.parameters() if p.requires_grad])}')

    def check_unfreeze(self, epoch):
        if self.o.args.unfreeze_gpt:
            self.GPT_unfreeze_layers(epoch)
        if self.o.args.unfreeze_vit:
            self.VIT_unfreeze_layers(epoch)

    def VIT_unfreeze_general_params(self):
        for param in self.vit_general_params:
            param.requires_grad = True

    def GPT_unfreeze_general_params(self):
        for param in self.gpt_general_params:
            param.requires_grad = True

    def GPT_unfreeze_layers(self, epoch):

        self.GPT_unfreeze_general_params()

        params = (self.o.args.mode, self.o.args.data)

        if params == ('unified', 'coco'):
            sched = self.o.train_config.decoder_unfreeze_unified
        elif params == ('cross', 'coco'):
            sched = self.o.train_config.decoder_unfreeze_cross
        elif params == ('unified', 'distance'):
            sched = self.o.train_config.decoder_unfreeze_unified_dist
        else:
            sched = self.o.train_config.decoder_unfreeze_cross_dist

        blocks_to_unfreeze = sched[epoch] if epoch in sched else []

        if len(blocks_to_unfreeze) > 0:

            print(f'\nUnfreezing GPT layers: {blocks_to_unfreeze}')
            for block_num in blocks_to_unfreeze:
                block = self.gpt_blocks[block_num]
                for layer in block:
                    for p in layer.parameters():
                        p.requires_grad = True

    def VIT_unfreeze_layers(self, epoch):
        self.VIT_unfreeze_general_params()

        params = (self.o.args.mode, self.o.args.data)

        if params == ('unified', 'coco'):
            sched = self.o.train_config.encoder_unfreeze_unified
        elif params == ('cross', 'coco'):
            sched = self.o.train_config.encoder_unfreeze_cross
        elif params == ('unified', 'distance'):
            sched = self.o.train_config.encoder_unfreeze_unified_dist
        else:
            sched = self.o.train_config.encoder_unfreeze_cross_dist

        blocks_to_unfreeze = sched[epoch] if epoch in sched else []

        if len(blocks_to_unfreeze) > 0:
            print(f'\nUnfreezing VIT layers: {blocks_to_unfreeze}')

            for block_num in blocks_to_unfreeze:
                block = self.vit_blocks[block_num]
                for p in block.parameters():
                    p.requires_grad = True

    def freeze_all_layers_all_models(self, trainable=False):

        all_params = []

        all_params.extend(self.gpt_general_params)
        all_params.extend(self.vit_general_params)
        all_params.extend(self.vit_blocks)

        gpt_layers = [[
            self.transformer.h[i].ln_1, self.transformer.h[i].ln_2,
            self.transformer.h[i].attn, self.transformer.h[i].mlp
        ] for i in range(self.o.model_config.depth)]
        for l in gpt_layers:
            all_params.extend(l)

        for layer in all_params:
            # Full Block
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = trainable
            # General Param
            else:
                layer.requires_grad = trainable

    def create_unified_input(self, token_ids, enriched_image):
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


        # Now we re-apply the image encoder positional encoding to the encoded image embeddings
        # I could have trained new position embeddings but re-use is simpler and more efficient
        # and my assumption is that this choice won't have a significant negative impact on my results.

        enriched_image = self.image_encoder.add_pos_embed(enriched_image)

        # The text tokens are appended to the image tokens, this ordering is important for masking in unified attention.
        unified_embeddings = torch.concat((enriched_image, text_embeddings), dim=1)

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

        enriched_image = self.image_encoder.forward(image)
        input_embeddings = self.create_unified_input(token_ids, enriched_image)

        hidden_state = input_embeddings

        # This should be encapsulated in a function that can be overridden in
        # a test model class that just returns hard-coded values.
        for i in range(self.m_cnfg.depth):
            hidden_state = self.transformer.h[i](hidden_state)

        hidden_state = self.transformer.ln_f(hidden_state)

        # If labels is not None, we are in training mode
        if labels is not None:
            # We need to get rid of the image sequences before we compute the logits
            hidden_state = self.remove_image_hidden_states(hidden_state)

            # lm_head layer projects the hidden state dimension 768 to the vocab dimension 50257
            lm_logits = self.lm_head(hidden_state)

            unrolled_labels = rearrange(labels, 'batch seq -> (batch seq)')
            unrolled_logits = rearrange(lm_logits, 'batch seq vocab -> (batch seq) vocab')

            loss = F.cross_entropy(unrolled_logits, unrolled_labels)
            return loss

        # This is inference mode text generation, where we predict the next token from last hidden state of the sequence
        last_hidden_state = self.remove_image_hidden_states(hidden_state)

        lm_logits = self.lm_head(last_hidden_state)
        return lm_logits

    # Note that
    def remove_image_hidden_states(self, hidden_state):
        hidden_state = hidden_state[:, NUM_IMAGE_TOKENS:, :]
        return hidden_state

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
    def from_pretrained(o):
        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        print(f'Loading pre-trained gpt2-small...')

        model = UnifiedAttentionModel(o)
        # We are going to replace the random values in our model's sd with pretrained weights
        sd = model.state_dict()

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

