import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from cyto_dl.nn.vits.blocks.cross_attention import Mlp


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1]))


class AttentionPooling(nn.Module):
    def __init__(
        self,
        encoder_dim,
        decoder_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """query from decoder (x), key and value from encoder (y)"""
        B, N, C = x.shape
        By, Ny, _ = y.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = (
            self.kv(y)
            .reshape(By, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop,
        )
        x = attn.transpose(1, 2).reshape(By, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionPoolingBlock(nn.Module):
    def __init__(
        self,
        encoder_dim,
        decoder_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(decoder_dim)
        self.pooling = AttentionPooling(
            encoder_dim,
            decoder_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm_layer(decoder_dim)
        mlp_hidden_dim = int(decoder_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=decoder_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x, y):
        """
        x: decoder feature; y: encoder feature (after layernorm)
        """
        x = x + self.pooling(self.norm1(x), y)
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim, input_dim, num_tokens):
        super().__init__()
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, input_dim))

        # self.pos_embedding = torch.nn.Parameter(torch.zeros(num_tokens + 1, 1, input_dim))
        self.pos_embedding = self.positionalencoding2d(input_dim, 32, 32)
        self.pos_embedding = rearrange(self.pos_embedding, "c h w -> (h w) 1 c")
        # self.pos_embedding= torch.cat([torch.zeros(1, 1, input_dim), self.pos_embedding], dim=0)
        self.pos_embedding = self.pos_embedding.cuda()

        self.latent_query = torch.nn.Parameter(torch.zeros(1, 1, latent_dim))

        self.encoder = AttentionPoolingBlock(
            encoder_dim=input_dim, decoder_dim=latent_dim, num_heads=4
        )
        self.decoder = AttentionPoolingBlock(
            encoder_dim=latent_dim, decoder_dim=input_dim, num_heads=4
        )

    def positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        import math

        if d_model % 4 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dimension (got dim={:d})".format(d_model)
            )
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0.0, width).unsqueeze(1)
        pos_h = torch.arange(0.0, height).unsqueeze(1)
        pe[0:d_model:2, :, :] = (
            torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        )
        pe[1:d_model:2, :, :] = (
            torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        )
        pe[d_model::2, :, :] = (
            torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )
        pe[d_model + 1 :: 2, :, :] = (
            torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )

        return pe

    def forward(self, input_features, forward_indexes):
        # features are  # masked tokens, batch, input_dim
        T, B, C = input_features.shape

        input_features = rearrange(input_features, "t b c -> b t c")

        # cross attend from latent query to features
        latent_features = self.encoder(self.latent_query, input_features)

        # recreate input features from latent features
        # forward_indexes = torch.cat(
        #     [torch.zeros(1, forward_indexes.shape[1]).to(forward_indexes), forward_indexes + 1],
        #     dim=0,
        # )
        # positional_embeddings = take_indexes(self.pos_embedding.expand(-1, B, -1), forward_indexes)
        # positional_embeddings= positional_embeddings[:T]

        # skip index taking for now
        positional_embeddings = self.pos_embedding.expand(-1, B, -1)
        #

        output_features = self.mask_token.expand(T, B, -1)
        output_features = output_features + positional_embeddings
        output_features = rearrange(output_features, "t b c -> b t c")

        # fill in masked regions with keys/values from latent features
        output_features = self.decoder(output_features, latent_features)

        output_features = rearrange(output_features, "b t c -> t b c")
        return output_features


if __name__ == "__main__":
    # test
    num_tokens = 1024
    encoder_dim = 128
    model = AttentionAutoencoder(128, encoder_dim, num_tokens).cuda()
    loss_fn = torch.nn.MSELoss()

    print("N model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    import numpy as np
    from aicsimageio import AICSImage
    from aicsimageio.writers import OmeTiffWriter

    img = (
        AICSImage(
            "//allen/aics/assay-dev/MicroscopyData/Leveille/2023/20230720/2023-07-20/AD00004745_20230720_AICS13_L01-01.czi/AD00004745_20230720_AICS13_L01-01_AcquisitionBlock1.czi/AD00004745_20230720_AICS13_L01-01_AcquisitionBlock1_pt1.czi"
        )
        .get_image_dask_data("ZYX")
        .compute()
    )
    img = img.max(0)
    img = (img - img.mean()) / img.std()
    img = torch.from_numpy(img).cuda().unsqueeze(0).unsqueeze(0).float()

    conv_size = 8
    weight = torch.ones(encoder_dim, 1, conv_size, conv_size) / conv_size**2

    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    for i in range(10000):
        start = np.random.choice(range(800))
        crop = img[:, :, start : start + 256, start : start + 256]
        crop = torch.nn.functional.conv2d(crop, weight.float().cuda(), stride=conv_size, padding=0)
        crop = rearrange(crop, "b c y x -> (y x) b c")
        forward_indexes = torch.arange(num_tokens).unsqueeze(1).cuda()
        output_features = model(crop, forward_indexes)

        loss = loss_fn(output_features, crop)
        loss.backward()
        opt.step()
        opt.zero_grad()

        if i % 100 == 0:
            print(loss.item(), crop.shape)
    crop = rearrange(crop, "(y x) b c -> b c y x", y=256 // conv_size)
    output_features = rearrange(output_features, "(y x) b c -> b c y x", y=256 // conv_size)
    OmeTiffWriter.save(
        uri="//allen/aics/assay-dev/users/Benji/input.tif",
        data=crop.detach().cpu().numpy().squeeze(),
    )
    OmeTiffWriter.save(
        uri="//allen/aics/assay-dev/users/Benji/output.tif",
        data=output_features.detach().cpu().numpy().squeeze(),
    )
