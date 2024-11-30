
import torch
import torch.nn as nn
from timm import create_model

class ImageEncoder(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.config = config

        vit = create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0
        )

        self.blocks = nn.ModuleList([vit.blocks[i] for i in range(config.depth)])
        self.vit_patch_embed = vit.patch_embed
        self.vit_pos_embed = vit.pos_embed
        self.vit_cls_token = vit.cls_token
        self.pos_drop = nn.Dropout(p=0.)

    def pos_embed(self, x):
        pos_embed = self.vit_pos_embed
        x = torch.cat((self.vit_cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)

    def forward(self, image):

        # "Tokenizes" the image into 16x16 patches
        patch_emb = self.patch_embed(image)
        pos_patch_emb = self.pos_embed(patch_emb)

        hidden_state = pos_patch_emb

        for i in range(self.config.depth):
            hidden_state = self.vision_blocks[i](hidden_state)

        return hidden_state