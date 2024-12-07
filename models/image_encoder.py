import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import torch
import torch.nn as nn
from timm import create_model

class ImageEncoder(nn.Module):
    def __init__(self, o):
        super().__init__()
        self.args = o.args
        self.config = o.model_config

        print(f'Loading pre-trained vit_base_patch16_224...')
        vit = create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0
        )

        self.blocks = nn.ModuleList([vit.blocks[i] for i in range(self.config.depth)])
        self.patch_embed = vit.patch_embed
        self.pos_embed = vit.pos_embed
        self.cls_token = vit.cls_token
        self.pos_drop = nn.Dropout(p=0.)

        # At this point we only have defined VIT params so this will only print those
        print(f'Total params VIT: {sum([p.numel() for p in self.parameters() if p.requires_grad])}')

    def pos_embed_with_cls(self, x):
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def add_pos_embed(self, image):
        return self.pos_embed + image

    def create_image_input(self, image):
        patch_emb = self.patch_embed(image)
        pos_patch_emb = self.pos_embed_with_cls(patch_emb)
        return pos_patch_emb

    def forward(self, image):




        hidden_state = self.create_image_input(image)

        for i in range(self.config.depth):
            hidden_state = self.blocks[i](hidden_state)

        return hidden_state