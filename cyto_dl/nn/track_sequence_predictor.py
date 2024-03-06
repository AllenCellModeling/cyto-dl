import math

import torch
from einops import rearrange
from monai.networks.nets import Regressor
from timm.models.vision_transformer import Block


class TrackClassifier(torch.nn.Module):
    """Transformer that encodes images in a sequence as tokens and classifies each of them."""

    def __init__(
        self,
        patch_size,
        pos_embedding_length=120,
        emb_dim=128,
        num_layer=8,
        num_head=4,
        num_classes=2,
    ) -> None:
        super().__init__()

        self.register_buffer("pos_embedding", positionalencoding1d(emb_dim, pos_embedding_length))
        self.pos_embedding.requires_grad = False

        self.image_encoder = Regressor(
            in_shape=patch_size, out_shape=emb_dim, channels=[8, 16, 32], strides=[2, 2, 2]
        )

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.classifier = torch.nn.Linear(emb_dim, num_classes)

    def forward(self, img):
        img = img.as_tensor()
        # move track length to front to embed images individually
        img = rearrange(img, "b track_len h w -> track_len b h w")

        # returns  token x batch x embedding dim
        patches = self.image_encoder(img).unsqueeze(1)

        # interpolate positional embedding to match track length
        pe = torch.nn.functional.interpolate(
            self.pos_embedding, size=patches.shape[0], mode="linear"
        )
        pe = rearrange(pe, "emb_dim 1 tokens -> tokens 1 emb_dim")

        patches = patches + pe

        patches = self.layer_norm(self.transformer(patches))

        patches = self.classifier(patches)
        patches = rearrange(patches, "tokens batch emb_dim -> batch tokens emb_dim")

        return patches


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with " "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return rearrange(pe, "tokens emb_dim -> emb_dim 1 tokens")
