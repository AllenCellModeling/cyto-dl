import torch
from timm.models.vision_transformer import Block
from monai.networks.nets import Regressor
from einops import rearrange

    

class TrackClassifier(torch.nn.Module):
    def __init__(
        self,
        patch_size,
        max_track_length,
        emb_dim=128,
        num_layer=8,
        num_head=4,
        num_classes = 4,
    ) -> None:
        super().__init__()

        self.register_buffer('pos_embedding', positionalencoding1d(emb_dim, max_track_length))
        self.pos_embedding.requires_grad = False

        self.image_encoder = Regressor(in_shape=patch_size, out_shape = emb_dim, channels = [8, 16, 32], strides = [2,2,2])

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.classifier = torch.nn.Linear(emb_dim, num_classes)

        self.max_track_length = max_track_length

    def forward(self, img):
        img = img.as_tensor()
        # assume batch size 1 so we don't have to pad tracks
        # embed each image in track
        img = rearrange(img, 'b c h w -> c b  h w')
        # return  t b c
        patches = self.image_encoder(img).unsqueeze(1)
        patches = patches + self.pos_embedding[:len(patches)]

        patches = self.layer_norm(self.transformer(patches))

        patches = self.classifier(patches)
        patches = rearrange(patches, 't b c -> b t c')

        return patches


import math
def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)
