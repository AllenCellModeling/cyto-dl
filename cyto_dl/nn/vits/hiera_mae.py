# modified from https://github.com/IcarusWizard/MAE/blob/main/model.py#L124

from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Block
from monai.networks.blocks import UnetOutBlock, UnetResBlock, UpSample


from cyto_dl.nn.vits.blocks.masked_unit_attention import HieraBlock

from cyto_dl.nn.vits.blocks.patchify_hiera import PatchifyHiera
from cyto_dl.nn.vits.mae import MAE_Decoder
from cyto_dl.nn.vits.cross_mae import CrossMAE_Decoder



class SpatialMerger(nn.Module):
    def __init__(self, downsample_factor, in_dim, out_dim):
        super().__init__()
        self.downsample_factor = downsample_factor
        conv = nn.Conv3d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0,
            bias=False,
        )

        tokens2img = Rearrange(
            "b n_mu (z y x) c -> (b n_mu) c z y x", z=downsample_factor[0], y=downsample_factor[1], x=downsample_factor[2]
        )
        self.model = nn.Sequential(
            tokens2img,
            conv
        )
    
    def forward(self, x):
        b, n_mu, _, _ = x.shape
        x = self.model(x)
        x = rearrange(x, "(b n_mu) c z y x -> b n_mu (z y x) c", b=b, n_mu=n_mu)
        return x

class HieraEncoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: List[int],
        num_mask_units: List[int],
        architecture: List[Dict],
        emb_dim: int = 64,
        spatial_dims: int = 3,
        patch_size: List[int] = (16, 16, 16),
        mask_ratio: Optional[float] = 0.75,
        context_pixels: Optional[List[int]] = [0, 0, 0],
        save_layers: Optional[bool] = True,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int]
            Number of patches in each dimension
        spatial_dims: int
            Number of spatial dimensions
        base_patch_size: List[int]
            Size of each patch
        emb_dim: int
            Dimension of embedding
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
        input_channels: int
            Number of input channels
        weight_intermediates: bool
            Whether to output linear combination of intermediate layers as final output like CrossMAE
        """
        super().__init__()
        self.save_layers=  save_layers
        self.patchify = PatchifyHiera(
            patch_size, num_patches, mask_ratio, num_mask_units, emb_dim, spatial_dims, context_pixels
        )

        patches_per_mask_unit = np.array(num_patches) // np.array(num_mask_units)
        self.final_dim = emb_dim * (2**len(architecture))

        self.save_block_idxs = []
        self.save_block_dims = []
        self.spatial_mergers = torch.nn.ParameterDict({})
        transformer = []
        num_blocks = 0
        for stage_num, stage in enumerate(architecture):
            # use mask unit attention until first layer that uses self attention
            if stage.get('self_attention', False):
                break
            print(f"Stage: {stage_num}")
            for block in range(stage["repeat"]):
                is_last = block == stage["repeat"] - 1
                # do spatial pooling within mask unit on last block of stage
                q_stride = stage['q_stride'] if is_last else [1] * spatial_dims

                # double embedding dimension in last block of stage
                dim_in = emb_dim * (2**stage_num)
                dim_out = dim_in if not is_last else dim_in * 2
                print(f"\tBlock {block}:\t\tdim_in: {dim_in}, dim_out: {dim_out}, num_heads: {stage['num_heads']}, q_stride: {q_stride}, patches_per_mask_unit: {patches_per_mask_unit}")
                transformer.append(
                    HieraBlock(
                        dim=dim_in,
                        dim_out=dim_out,
                        heads=stage['num_heads'],
                        q_stride = q_stride,
                        patches_per_mask_unit = patches_per_mask_unit,
                    )
                )
                if is_last:
                    # save the block before the spatial pooling unless it's the final stage
                    save_block = num_blocks -1 if stage_num < len(architecture) - 1 else num_blocks
                    self.save_block_idxs.append(save_block)
                    self.save_block_dims.append(dim_out)
                    
                    # create a spatial merger for combining tokens pre-downsampling, last stage doesn't need merging since it has expected num channels, spatial shape
                    self.spatial_mergers[f'block_{save_block}'] = SpatialMerger(patches_per_mask_unit, dim_in, self.final_dim) if stage_num < len(architecture) - 1 else torch.nn.Identity()

                    # at end of each layer, patches per mask unit is reduced as we pool spatially
                    patches_per_mask_unit  = patches_per_mask_unit //  np.array(stage['q_stride'])
                num_blocks += 1
        self.mask_unit_transformer = torch.nn.Sequential(*transformer)

        self.self_attention_transformer = torch.nn.Sequential(
                *[Block(self.final_dim, stage['num_heads']) for _ in range(stage['repeat'])]
            )

        self.layer_norm = torch.nn.LayerNorm(self.final_dim)

    def forward(self, img):
        patches, mask, forward_indexes, backward_indexes = self.patchify(img)

        # mask unit attention
        mask_unit_embeddings = 0.0
        save_layers = []
        for i, block in enumerate(self.mask_unit_transformer):
            patches = block(patches)
            if i in self.save_block_idxs:
                mask_unit_embeddings += self.spatial_mergers[f'block_{i}'](patches)
                if self.save_layers:
                    save_layers.append(patches)

        # combine mask units and tokens for full self attention transformer
        mask_unit_embeddings = rearrange(mask_unit_embeddings, "b n_mu t c -> b (n_mu t) c")
        mask_unit_embeddings = self.self_attention_transformer(mask_unit_embeddings)
        mask_unit_embeddings= self.layer_norm(mask_unit_embeddings)

        return mask_unit_embeddings, mask, forward_indexes, backward_indexes, save_layers


class HieraMAE(torch.nn.Module):
    def __init__(
        self,
        architecture,
        spatial_dims: int = 3,
        num_patches: Optional[List[int]] = [2, 32, 32],
        num_mask_units: Optional[List[int]] = [2, 12, 12],
        patch_size: Optional[List[int]] = [16, 16, 16],
        emb_dim: Optional[int] = 64,
        decoder_layer: Optional[int] = 4,
        decoder_head: Optional[int] = 8,
        decoder_dim: Optional[int] = 192,
        mask_ratio: Optional[int] = 0.75,
        context_pixels: Optional[List[int]] = [0, 0, 0],
        use_crossmae: Optional[bool] = False,
    ) -> None:
        """
        Parameters
        ----------
        """
        super().__init__()
        assert spatial_dims in (2, 3), "Spatial dims must be 2 or 3"

        if isinstance(num_patches, int):
            num_patches = [num_patches] * spatial_dims
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial_dims

        assert len(num_patches) == spatial_dims, "num_patches must be of length spatial_dims"
        assert (
            len(patch_size) == spatial_dims
        ), "patch_size must be of length spatial_dims"

        self.mask_ratio = mask_ratio

        self.encoder = HieraEncoder(
            num_patches=num_patches,
            num_mask_units=num_mask_units,
            architecture=architecture,
            emb_dim=emb_dim,
            spatial_dims=spatial_dims,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            context_pixels=context_pixels,
        )
        # "patches" to the decoder are actually mask units, so num_patches is num_mask_units, patch_size is mask unit size
        mask_unit_size = (np.array(num_patches) * np.array(patch_size))/np.array(num_mask_units)

        decoder_class = MAE_Decoder
        if use_crossmae:
            decoder_class = CrossMAE_Decoder

        self.decoder = decoder_class(
            num_patches=num_mask_units,
            spatial_dims=spatial_dims,
            base_patch_size=mask_unit_size.astype(int),
            enc_dim=self.encoder.final_dim,
            emb_dim=decoder_dim,
            num_layer=decoder_layer,
            num_head=decoder_head,
            has_cls_token=False
        )

    def forward(self, img):
        features, mask, forward_indexes, backward_indexes, save_layers = self.encoder(img)
        features = rearrange(features, "b t c -> t b c")
        predicted_img = self.decoder(features, forward_indexes, backward_indexes)
        return predicted_img, mask


class HieraSeg(torch.nn.Module):
    def __init__(
        self,
        encoder_ckpt,
        architecture,
        spatial_dims: int = 3,
        n_out_channels: int = 6,
        num_patches: Optional[List[int]] = [2, 32, 32],
        num_mask_units: Optional[List[int]] = [2, 12, 12],
        patch_size: Optional[List[int]] = [16, 16, 16],
        emb_dim: Optional[int] = 64,
        context_pixels: Optional[List[int]] = [0, 0, 0],
    ) -> None:
        """
        Parameters
        ----------
        """
        super().__init__()
        assert spatial_dims in (2, 3), "Spatial dims must be 2 or 3"

        if isinstance(num_patches, int):
            num_patches = [num_patches] * spatial_dims
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial_dims

        assert len(num_patches) == spatial_dims, "num_patches must be of length spatial_dims"
        assert (
            len(patch_size) == spatial_dims
        ), "patch_size must be of length spatial_dims"


        self.encoder = HieraEncoder(
            num_patches=num_patches,
            num_mask_units=num_mask_units,
            architecture=architecture,
            emb_dim=emb_dim,
            spatial_dims=spatial_dims,
            patch_size=patch_size,
            mask_ratio=0,
            context_pixels=context_pixels,
            save_layers=True
        )
        # model = torch.load(encoder_ckpt, map_location="cuda:0")
        # enc_state_dict = {
        #     k.replace("backbone.encoder.", ""): v
        #     for k, v in model["state_dict"].items()
        #     if "encoder" in k and "intermediate" not in k
        # }
        # self.encoder.load_state_dict(enc_state_dict, strict=False)
        # for name, param in self.encoder.named_parameters():
        #     # allow different weighting of internal activations for finetuning
        #     param.requires_grad = False

        # "patches" to the decoder are actually mask units, so num_patches is num_mask_units, patch_size is mask unit size
        mask_unit_size = ((np.array(num_patches) * np.array(patch_size))/np.array(num_mask_units)).astype(int)

        project_dim = np.prod(mask_unit_size)*16
        head = torch.nn.Linear(self.encoder.final_dim, project_dim)
        norm = torch.nn.LayerNorm(project_dim)
        patch2img =  Rearrange(
            "(n_patch_z n_patch_y n_patch_x) b (c patch_size_z patch_size_y patch_size_x) ->  b c (n_patch_z patch_size_z) (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
            n_patch_z=num_mask_units[0],
            n_patch_y=num_mask_units[1],
            n_patch_x=num_mask_units[2],
            patch_size_z=mask_unit_size[0],
            patch_size_y=mask_unit_size[1],
            patch_size_x=mask_unit_size[2],
        )
        self.patch2img = torch.nn.Sequential(head, norm, patch2img)

        self.upsample = torch.nn.Sequential(
            *[
                UpSample(
                    spatial_dims=spatial_dims,
                    in_channels=16,
                    out_channels=16,
                    scale_factor=[2.6134, 2.5005, 2.5005],
                    mode="nontrainable",
                    interp_mode="trilinear",
                ),
                UnetResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=16,
                    out_channels=16,
                    stride=1,
                    kernel_size=3,
                    norm_name="INSTANCE",
                    dropout=0,
                ),
                UnetOutBlock(
                    spatial_dims=spatial_dims,
                    in_channels=16,
                    out_channels=n_out_channels,
                    dropout=0,
                ),
            ]
        )

    def forward(self, img):
        breakpoint()
        features, _, _, _, save_layers = self.encoder(img)
        features = rearrange(features, "b t c -> t b c")
        predicted_img = self.patch2img(features)
        predicted_img = self.upsample(predicted_img)
        return predicted_img
    



class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
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

    def forward(self, x, y, mask):
        B, N, C = x.shape
        Ny = y.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = (
            self.kv(y)
            .reshape(B, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop,
            attn_mask=mask>0.5 if mask is not None else None,
        )
        x = attn.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
from timm.models.vision_transformer import Attention

class Mask2FormerBlock(nn.Module):
    def __init__(
        self,
        encoder_dim,
        decoder_dim,
        num_heads,
        num_patches,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(decoder_dim)

        # TODO add positional embedding and scale embedding to image features
        self.scale_positional_embedding = nn.Parameter(torch.zeros(1, num_patches, encoder_dim))

        self.self_attn_block = Attention(
            dim=decoder_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.cross_attn = CrossAttention(
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

    def forward(self, x, y, mask):
        """
        x: query features, y: image features, mask: previous mask prediction
        """
        x = self.norm1(x + self.cross_attn(x, y, mask))
        x = x + self.self_attn_block(x)
        x = x + self.mlp(self.norm2(x))
        return x

class HieraMask2Former(torch.nn.Module):
    def __init__(
        self,
        encoder_ckpt,
        architecture,
        spatial_dims: int = 3,
        num_queries: int = 50,
        num_patches: Optional[List[int]] = [2, 32, 32],
        num_mask_units: Optional[List[int]] = [2, 12, 12],
        patch_size: Optional[List[int]] = [16, 16, 16],
        emb_dim: Optional[int] = 64,
        context_pixels: Optional[List[int]] = [0, 0, 0],
    ) -> None:
        """
        Parameters
        ----------
        """
        super().__init__()
        assert spatial_dims in (2, 3), "Spatial dims must be 2 or 3"

        if isinstance(num_patches, int):
            num_patches = [num_patches] * spatial_dims
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial_dims

        assert len(num_patches) == spatial_dims, "num_patches must be of length spatial_dims"
        assert (
            len(patch_size) == spatial_dims
        ), "patch_size must be of length spatial_dims"
        self.num_mask_units = num_mask_units

        self.encoder = HieraEncoder(
            num_patches=num_patches,
            num_mask_units=num_mask_units,
            architecture=architecture,
            emb_dim=emb_dim,
            spatial_dims=spatial_dims,
            patch_size=patch_size,
            mask_ratio=0,
            context_pixels=context_pixels,
        )
        model = torch.load(encoder_ckpt, map_location="cuda:0")
        enc_state_dict = {
            k.replace("backbone.encoder.", ""): v
            for k, v in model["state_dict"].items()
            if "encoder" in k and "intermediate" not in k
        }
        self.encoder.load_state_dict(enc_state_dict, strict=False)
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        # "patches" to the decoder are actually mask units, so num_patches is num_mask_units, patch_size is mask unit size
        mask_unit_size = ((np.array(num_patches) * np.array(patch_size))/np.array(num_mask_units)).astype(int)

        self.instance_queries = torch.nn.Parameter(torch.zeros(1, num_queries, self.encoder.final_dim))
        self.instance_queries_pos_emb = torch.nn.Parameter(torch.zeros(1, num_queries, self.encoder.final_dim))

        q_strides = [np.array(stage['q_stride']) for stage in architecture if stage.get('q_stride', False)]
        patches_per_mask_unit = [np.array(num_patches) // np.array(num_mask_units)]
        for qs in q_strides:            
            patches_per_mask_unit.append(patches_per_mask_unit[-1] // qs)
        patches_per_mask_unit.reverse()
        patches_per_mask_unit[0] = np.array([1,1,1])
        self.patches_per_mask_unit = patches_per_mask_unit

        # TODO each block should have a different embedding dimension
        self.transformer = torch.nn.ModuleList([Mask2FormerBlock(encoder_dim = self.encoder.save_block_dims[i], decoder_dim = 128, num_neads = 4, num_patches = patches_per_mask_unit[i] * num_mask_units) for i in range(len(patches_per_mask_unit))])



    def forward(self, img):
        breakpoint()
        #features are b x t x c
        features, _, _, _, save_layers = self.encoder(img)
        save_layers.append(features.unsqueeze(2))
        save_layers.reverse()
        # start with lowest resolution
        # first mask should be prediction from query features alone
        mask = None
        for layer, ppmu in zip(save_layers, self.patches_per_mask_unit):
            layer = rearrange(layer, 'b n_mu mu_dims c -> b (n_mu mu_dims) c')

            layer = self.transformer(layer, mask, self.instance_queries, self.instance_queries_pos_emb)

            # cross attention provides one mask per query
            # self attention refines mask
            # repeat for each block

            # rearrange to mask TODO make this account for havingn_queries masks
            img_features = rearrange(layer, 'b (n_mu_z n_mu_y n_mu_x) (patches_per_mu_z patches_per_mu_y patches_per_mu_x) c -> b c (n_mu_z patches_per_mu_z) (n_mu_y patches_per_mu_y) (n_mu_x patches_per_mu_x)', n_mu_z=self.num_mask_units[0], n_mu_y=self.num_mask_units[1], n_mu_x=self.num_mask_units[2], patches_per_mu_z=ppmu[0], patches_per_mu_y=ppmu[1], patches_per_mu_x=ppmu[2])

            # upsample to next resolution
            mask = F.interpolate(mask, scale_factor=ppmu, mode='nearest')

        return predicted_img