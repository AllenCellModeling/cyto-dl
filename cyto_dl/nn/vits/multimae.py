from typing import Dict, List, Optional

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch.distributions import Dirichlet

from cyto_dl.nn.vits.blocks import CrossAttentionBlock, IntermediateWeigher, Patchify
from cyto_dl.nn.vits.utils import take_indexes


class MultiMAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        input_types: Dict[str, List[str]],
        num_patches: List[int],
        spatial_dims: int = 3,
        base_patch_size: List[int] = (16, 16, 16),
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 12,
        num_head: Optional[int] = 3,
        context_pixels: Optional[List[int]] = [0, 0, 0],
        input_channels: Optional[int] = 1,
        n_intermediate_weights: Optional[int] = -1,
        alpha: Optional[float] = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        tasks: Dict[str, List[str]]
            List of tasks to encode in format  {category1: [task1, ...]} where category could be the modality (e.g. flourescent image, brightfield) and task could be the structure present in the flourescent image
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
        n_intermediate_weights: int
            Number of intermediate weights to use in the intermediate weigher
        alpha: float
            Alpha parameter for the Dirichlet distribution
        """
        super().__init__()
        self.alpha = alpha
        self.task2category = {
            task: category for category, tasks in input_types.items() for task in tasks
        }
        self.num_patches = np.asarray(num_patches)

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

        # create different patch embedding layer per-modality
        self.patchify = torch.nn.ModuleDict(
            {
                modality: Patchify(
                    base_patch_size,
                    emb_dim,
                    num_patches,
                    spatial_dims,
                    context_pixels,
                    input_channels,
                    tasks=tasks,
                )
                for modality, tasks in input_types.items()
            }
        )

        weight_intermediates = n_intermediate_weights > 0
        if weight_intermediates:
            self.transformer = torch.nn.ModuleList(
                [Block(emb_dim, num_head) for _ in range(num_layer)]
            )
        else:
            self.transformer = torch.nn.Sequential(
                *[Block(emb_dim, num_head) for _ in range(num_layer)]
            )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.intermediate_weighter = (
            IntermediateWeigher(num_layer, emb_dim, n_intermediate_weights)
            if weight_intermediates
            else None
        )
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)

    def generate_task_mask_ratios(self, tasks, mask_ratio):
        # we might want to make the mask ratio vary across elements of the batch? as long as total # tokens is fixed wecan concat across batches!
        tasks_present = [k for k in tasks.keys() if k in self.task2category]
        B = tasks[tasks_present[0]].shape[0]

        patches_per_img = np.prod(self.num_patches)
        total_visible_tokens = int(mask_ratio * patches_per_img * len(tasks_present) * B)
        task_sampling_ratios = (
            Dirichlet(torch.Tensor([self.alpha] * len(tasks_present)))
            .sample((1,))
            .squeeze()
            .tolist()
        )
        task_mask_ratios = {
            task: int(total_visible_tokens * mask_ratio)
            for task, mask_ratio in zip(tasks_present, task_sampling_ratios)
        }
        return task_mask_ratios

    def forward(self, tasks, mask_ratio):
        # tasks are {task1: tensor, task2: tensor, ...}
        task_mask_ratios = self.generate_task_mask_ratios(tasks, mask_ratio)

        meta = {}
        task_tokens = []
        for task_name, img in tasks.items():
            if task_name not in self.task2category:
                continue
            category = self.task2category[task_name]
            # mask and add posembed and task embed tokens
            patches, mask, forward_indexes, backward_indexes = self.patchify[category](
                img, mask_ratio=task_mask_ratios[task_name], task=task_name
            )

            # patches are tbc order
            task_tokens.append(patches)
            meta[task_name] = {
                "mask": mask,
                "forward_indexes": forward_indexes,
                "backward_indexes": backward_indexes,
                "n_tokens": len(patches),
            }

        task_tokens = torch.cat(task_tokens, dim=0)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, "t b c -> b t c")

        if self.intermediate_weighter is not None:
            intermediates = [patches]
            for block in self.transformer:
                patches = block(patches)
                intermediates.append(patches)
            features = self.layer_norm(self.intermediate_weighter(intermediates))
            features = rearrange(features, "n b t c -> n t b c")
        else:
            features = self.layer_norm(self.transformer(patches))
            features = rearrange(features, "b t c -> t b c")

        return features, meta


class OutputAdapter(torch.nn.Module):
    def __init__(
        self,
        base_patch_size: List[int],
        enc_dim: int,
        emb_dim: int,
        num_layer: int,
        num_head: int,
    ) -> None:
        super().__init__()
        self.transformer = torch.nn.ParameterList(
            [
                CrossAttentionBlock(
                    encoder_dim=emb_dim,
                    decoder_dim=emb_dim,
                    num_heads=num_head,
                )
                for _ in range(num_layer)
            ]
        )

        self.decoder_norm = torch.nn.LayerNorm(emb_dim)
        self.projection_norm = torch.nn.LayerNorm(emb_dim)
        self.projection = torch.nn.Linear(enc_dim, emb_dim)
        self.head = torch.nn.Linear(emb_dim, torch.prod(torch.as_tensor(base_patch_size)))

    def project(self, features):
        return self.projection_norm(self.projection(features))

    def forward(self, masked, visible):
        for transformer in self.transformer:
            masked = transformer(masked, visible)
        return masked

    def output(self, masked):
        return self.head(self.decoder_norm(masked))


class MultiMAE_Decoder(torch.nn.Module):
    """Decoder inspired by [CrossMAE](https://crossmae.github.io/) where masked tokens only attend
    to visible tokens."""

    def __init__(
        self,
        input_types: Dict[str, List[str]],
        num_patches: List[int],
        spatial_dims: int = 3,
        base_patch_size: Optional[List[int]] = [4, 8, 8],
        enc_dim: Optional[int] = 768,
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 4,
        num_head: Optional[int] = 3,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int]
            Number of patches in each dimension
        base_patch_size: Tuple[int]
            Size of each patch
        enc_dim: int
            Dimension of encoder
        emb_dim: int
            Dimension of embedding
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        """
        super().__init__()
        self.task2category = {
            task: category for category, tasks in input_types.items() for task in tasks
        }

        all_tasks = []
        for tasks in input_types.values():
            tasks.extend(tasks)

        # decoder transformer for each task
        self.output_adapters = torch.nn.ModuleDict(
            {
                modality: OutputAdapter(base_patch_size, enc_dim, emb_dim, num_layer, num_head)
                for modality in input_types
            }
        )
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(np.prod(num_patches) + 1, 1, emb_dim))
        self.task_embedding = torch.nn.ParameterDict(
            {task: torch.nn.Parameter(torch.zeros(1, 1, emb_dim)) for task in all_tasks}
        )

        if spatial_dims == 3:
            self.patch2img = Rearrange(
                "(n_patch_z n_patch_y n_patch_x) b (c patch_size_z patch_size_y patch_size_x) ->  b c (n_patch_z patch_size_z) (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
                n_patch_z=num_patches[0],
                n_patch_y=num_patches[1],
                n_patch_x=num_patches[2],
                patch_size_z=base_patch_size[0],
                patch_size_y=base_patch_size[1],
                patch_size_x=base_patch_size[2],
            )
        elif spatial_dims == 2:
            self.patch2img = Rearrange(
                "(n_patch_y n_patch_x) b (c patch_size_y patch_size_x) ->  b c (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
                n_patch_y=num_patches[0],
                n_patch_x=num_patches[1],
                patch_size_y=base_patch_size[0],
                patch_size_x=base_patch_size[1],
            )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def add_task_pos_emb(self, features, forward_indices, backward_indices, task):
        T, B, C = features.shape
        # fill in masked regions
        features = torch.cat(
            [
                features,
                self.mask_token.expand(
                    backward_indices.shape[0] - features.shape[0], features.shape[1], -1
                ),
            ],
            dim=0,
        )
        # unshuffle to original positions for positional embedding
        features = take_indexes(features, backward_indices)
        features = features + self.pos_embedding + self.task_embedding[task]

        # reshuffle
        features = take_indexes(features, forward_indices)
        return {task: {"visible": features[:T], "masked": features[T:]}}

    def forward(self, features, meta):
        # paper uses task + mask tokens as cross attention queries to everything else, can we do cross mae by just doing mask tokens to everything? then we don't have to separate everything out
        # they also don't remove the task tokens from the contextembeddings so the "cross attention" layer is also doing self attention to the visual tokens of each task basically

        # HACK TODO allow usage of multiple intermediate feature weights, this works when decoder is 0 layers
        features = features.squeeze(0)
        T, B, C = features.shape

        # get visible and masked tokens for all tasks
        features_with_embed = {}
        start = 1
        for task, task_meta in meta.items():
            end = start + meta[task]["n_tokens"]
            # include the cls token - TODO - determine if cls_token + task_embed is meaningful? if not, need to project cls token separately
            task_features = torch.cat([features[:1], features[start:end]], dim=0)
            task_features = self.output_adapters[task].project(task_features)
            features_with_embed.update(
                self.add_task_pos_emb(
                    task_features,
                    task_meta["forward_indexes"],
                    task_meta["backward_indexes"],
                    task,
                )
            )
            start = end

        # cross attention between masked tokens of each task and visible tokens of all tasks
        for task in meta:
            mask_tokens = features_with_embed[task]["masked"]
            visible_tokens = torch.cat([features_with_embed[t]["visible"] for t in meta], dim=0)

            mask_tokens = rearrange(mask_tokens, "t b c -> b t c")
            visible_tokens = rearrange(visible_tokens, "t b c -> b t c")

            mask_tokens = self.output_adapters[task](mask_tokens, visible_tokens)

            # noneed to remove cls token, it's a part of the visible tokens
            mask_tokens = rearrange(mask_tokens, "b t c -> t b c")

            # (npatches x npatches x npatches) b (emb dim) -> (npatches* npatches * npatches) b (z y x)
            mask_tokens = self.output_adapters[task].output(mask_tokens)

            # add back in visible/encoded tokens that we don't calculate loss on
            patches = torch.cat(
                [
                    torch.zeros((T - 1, B, patches.shape[-1]), requires_grad=False).to(patches),
                    patches,
                ],
                dim=0,
            )
            patches = take_indexes(patches, meta[task]["backward_indices"][1:] - 1)

            meta["reconstruction"] = self.patch2img(patches)
        return meta


class MultiMAE(torch.nn.Module):
    def __init__(
        self,
        input_types: Dict[str, List[str]],
        num_patches: List[int],
        spatial_dims: int = 3,
        base_patch_size: List[int] = (16, 16, 16),
        emb_dim: Optional[int] = 192,
        encoder_layer: Optional[int] = 12,
        encoder_head: Optional[int] = 3,
        decoder_layer: Optional[int] = 4,
        decoder_head: Optional[int] = 3,
        decoder_dim: Optional[int] = 64,
        context_pixels: Optional[List[int]] = [0, 0, 0],
        input_channels: Optional[int] = 1,
        alpha: Optional[float] = 1.0,
        mask_ratio: Optional[float] = 0.75,
    ) -> None:
        """
        Parameters
        ----------
        tasks: Dict[str, List[str]]
            List of tasks to encode in format  {category1: [task1, ...]} where category could be the modality (e.g. flourescent image, brightfield) and task could be the structure present in the flourescent image
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
        alpha: float
            Alpha parameter for the Dirichlet distribution
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = MultiMAE_Encoder(
            input_types,
            num_patches,
            spatial_dims,
            base_patch_size,
            emb_dim,
            encoder_layer,
            encoder_head,
            context_pixels,
            input_channels,
            decoder_layer,
            alpha,
        )
        self.decoder = MultiMAE_Decoder(
            input_types=input_types,
            num_patches=num_patches,
            spatial_dims=spatial_dims,
            base_patch_size=base_patch_size,
            enc_dim=emb_dim,
            emb_dim=decoder_dim,
            num_layer=decoder_layer,
            num_head=decoder_head,
        )

    def forward(self, batch):
        features, meta = self.encoder(batch, self.mask_ratio)
        return self.decoder(features, meta)
