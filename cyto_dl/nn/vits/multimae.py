from typing import List, Optional, Dict

import numpy as np
import torch
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from cyto_dl.nn.vits.blocks import IntermediateWeigher, Patchify
from torch.distributions import Dirichlet

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
        self.alpha= alpha
        self.task2category = {task: category for category, tasks in input_types.items() for task in tasks}
        self.num_patches = np.asarray(num_patches)
        
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

        # create different patch embedding layer per-modality
        self.patchify = torch.nn.ModuleDict({
            modality: Patchify(
                base_patch_size, emb_dim, num_patches, spatial_dims, context_pixels, input_channels, tasks=tasks
            )
            for modality, tasks in input_types.items()
        })
 
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
        tasks_present= [k for k in tasks.keys() if k in self.task2category]
        assert 1-1/len(tasks_present) <= mask_ratio < 1, f"mask ratio must be between {1/len(tasks_present)} and 1"
        B = tasks[tasks_present[0]].shape[0]

        visible_tokens_per_img =  int((1-mask_ratio) *np.prod(self.num_patches))
        total_visible_tokens = visible_tokens_per_img * len(tasks_present) # don't include batch in this yet

        task_sampling_ratios = Dirichlet(torch.Tensor([self.alpha]*len(tasks_present))).sample((1,)).squeeze()
        visible_tokens_per_task = (total_visible_tokens * task_sampling_ratios).int()
        # give leftover tokens to least sampled task
        leftover_tokens = total_visible_tokens - visible_tokens_per_task.sum()
        visible_tokens_per_task[torch.argmin(task_sampling_ratios)] += leftover_tokens

        visible_tokens_per_task = {task: int(t) for task, t in zip(tasks_present, visible_tokens_per_task)}
        return visible_tokens_per_task
        # task_mask_ratios = {task: int(t*tmr) for task, tmr in zip(tasks_present, task_sampling_ratios)}
        # return task_mask_ratios


    def forward(self, tasks, mask_ratio):
        # tasks are {task1: tensor, task2: tensor, ...}
        task_mask_ratios= self.generate_task_mask_ratios(tasks, mask_ratio)

        meta = {}
        task_tokens = []
        for task_name, img in tasks.items():
            if task_name not in self.task2category:
                continue
            category = self.task2category[task_name]
            # mask and add posembed and task embed tokens
            patches, mask, forward_indexes, backward_indexes = self.patchify[category](img, n_visible_patches = task_mask_ratios[task_name], task = task_name)

            # patches are tbc order
            task_tokens.append(patches)
            meta[task_name] = {
                'mask': mask,
                'forward_indices': forward_indexes,
                'backward_indices': backward_indexes,
                'n_tokens': len(patches)
            }

        task_tokens = torch.cat(task_tokens, dim=0)
        task_tokens = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), task_tokens], dim=0)
        task_tokens = rearrange(task_tokens, "t b c -> b t c")

        if self.intermediate_weighter is not None:
            intermediates = [task_tokens]
            for block in self.transformer:
                intermediates.append(block(intermediates[-1]))
            features = self.layer_norm(self.intermediate_weighter(intermediates))
            features = rearrange(features, "n b t c -> n t b c")
        else:
            features = self.layer_norm(self.transformer(patches))
            features = rearrange(features, "b t c -> t b c")

        return features, meta

class ProjectNorm(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.norm = torch.nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.norm(self.linear(x))

class MultiMAE_Decoder(torch.nn.Module):
    """Decoder inspired by [CrossMAE](https://crossmae.github.io/) where masked tokens only attend
    to visible tokens."""

    def __init__(
        self,
        input_types: Dict[str, List[str]],
        num_patches: List[int],
        enc_dim: Optional[int] = 192,
        emb_dim: Optional[int] = 256,
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
        """
        super().__init__()
        self.task2category = {task: category for category, tasks in input_types.items() for task in tasks}

        all_tasks = []
        for tasks in input_types.values():
            all_tasks.extend(tasks)

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        # add 1 for cls token
        self.pos_embedding = torch.nn.Parameter(torch.zeros(np.prod(num_patches) + 1, 1, emb_dim))
        self.task_embedding = torch.nn.ParameterDict({task: torch.nn.Parameter(torch.zeros(1, 1, emb_dim)) for task in all_tasks})
        self.project = torch.nn.ParameterDict({task: ProjectNorm(enc_dim, emb_dim) for task in all_tasks})


        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def _add_cls_token_to_indices(self, indices):
        return torch.cat(
            [torch.zeros(1, indices.shape[1]).to(indices), indices + 1],
            dim=0,
        )

    def add_task_pos_emb(self, features, forward_indices, backward_indices, task):
        # num weighted intermediate layers, num tokens, batch, embedding dimension
        N, T, B, C = features.shape

        backward_indices = self._add_cls_token_to_indices(backward_indices)
        forward_indices = self._add_cls_token_to_indices(forward_indices)

        # fill in masked regions
        features = torch.cat(
            [
                features,
                self.mask_token.expand(N,backward_indices.shape[0] - T, B, -1),
            ],
            dim=1,
        )

        # unshuffle to original positions for positional embedding
        features = take_indexes(features, backward_indices)
        features = features + self.pos_embedding + self.task_embedding[task]

        #reshuffle
        features = take_indexes(features, forward_indices)
        return {
            'visible_tokens':features[:, :T],
            # masked are the same (mask + pos + task) for all N (weighted input features)
            'masked_tokens':features[0, T:]
        }


    def forward(self, features, meta):
        # get visible and masked tokens for all tasks
        start = 1
        for task, task_meta in meta.items():
            end = start + meta[task]['n_tokens']
            # include the cls token - TODO - determine if cls_token + task_embed is meaningful? if not, need to project cls token separately
            task_features= torch.cat([features[:, :1], features[:, start:end]], dim=1)
            task_features = self.project[task](task_features)
            task_features = self.add_task_pos_emb(task_features, task_meta['forward_indices'], task_meta['backward_indices'], task)
            meta[task].update(task_features)
            start = end
        meta['visible_tokens'] = torch.cat([meta[t]['visible_tokens'] for t in meta ], dim =1)
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
        encoder_head: Optional[int] = 4,
        decoder_layer: Optional[int] = 4,
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
            enc_dim=emb_dim,
            emb_dim=decoder_dim,
        )

    def forward(self, batch):
        features, meta = self.encoder(batch, self.mask_ratio)
        return self.decoder(features, meta)


