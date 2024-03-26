# from typing import List, Optional

# import numpy as np
# import torch
# import torch.nn as nn
# from einops import rearrange
# from timm.models.layers import trunc_normal_
# from timm.models.vision_transformer import Block

# from cyto_dl.nn.vits.blocks import IntermediateWeigher, Patchify


# class MultiMAE_Encoder(torch.nn.Module):
#     def __init__(
#         self,
#         tasks: List[str],
#         num_patches: List[int],
#         spatial_dims: int = 3,
#         base_patch_size: List[int] = (16, 16, 16),
#         emb_dim: Optional[int] = 192,
#         num_layer: Optional[int] = 12,
#         num_head: Optional[int] = 3,
#         context_pixels: Optional[List[int]] = [0, 0, 0],
#         input_channels: Optional[int] = 1,
#         n_intermediate_weights: Optional[int] = -1,
#     ) -> None:
#         """
#         Parameters
#         ----------
#         tasks: List[str]
#             List of tasks to encode
#         num_patches: List[int]
#             Number of patches in each dimension
#         spatial_dims: int
#             Number of spatial dimensions
#         base_patch_size: List[int]
#             Size of each patch
#         emb_dim: int
#             Dimension of embedding
#         num_layer: int
#             Number of transformer layers
#         num_head: int
#             Number of heads in transformer
#         context_pixels: List[int]
#             Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
#         input_channels: int
#             Number of input channels
#         weight_intermediates: bool
#             Whether to output linear combination of intermediate layers as final output like CrossMAE
#         """
#         super().__init__()
#         self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
#         self.patchify = Patchify(
#             base_patch_size, emb_dim, num_patches, spatial_dims, context_pixels, input_channels
#         )
#         weight_intermediates = n_intermediate_weights > 0
#         if weight_intermediates:
#             self.transformer = torch.nn.ModuleList(
#                 [Block(emb_dim, num_head) for _ in range(num_layer)]
#             )
#         else:
#             self.transformer = torch.nn.Sequential(
#                 *[Block(emb_dim, num_head) for _ in range(num_layer)]
#             )

#         self.layer_norm = torch.nn.LayerNorm(emb_dim)

#         self.intermediate_weighter = (
#             IntermediateWeigher(num_layer, emb_dim, n_intermediate_weights)
#             if weight_intermediates
#             else None
#         )
#         self.task_embeddings = nn.ModuleDict(
#             {task: nn.Parameter(torch.randn(1, 1, emb_dim)) for task in tasks}
#         )
#         self.init_weight()

#     def init_weight(self):
#         trunc_normal_(self.cls_token, std=0.02)
#         for task in self.task_embeddings:
#             trunc_normal_(self.task_embeddings[task], std=0.02)


#     def forward(self, tasks):
#         ### Multi mae encoder
#         task_tokens_dict = {}
#         input_tokens = []
#         for task in input.tasks():
#             # mask, posembed, and task embed tokens
#             task_tokens = patchify(task, use_task_embedding=True)
#             task.tokens.append(task_tokens)
#             task_tokens_dict[task] = len(task_tokens)
#         task_tokens = torch.cat(global_token, task_tokens)
#         enc = self.encode(task_tokens)

#         task_mask_ratios= self.generate_mask_ratios()

#         patches, mask, forward_indexes, backward_indexes = self.patchify(img, mask_ratio)
#         patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
#         patches = rearrange(patches, "t b c -> b t c")

#         if self.intermediate_weighter is not None:
#             intermediates = [patches]
#             for block in self.transformer:
#                 patches = block(patches)
#                 intermediates.append(patches)
#             features = self.layer_norm(self.intermediate_weighter(intermediates))
#             features = rearrange(features, "n b t c -> n t b c")
#         else:
#             features = self.layer_norm(self.transformer(patches))
#             features = rearrange(features, "b t c -> t b c")
#         if mask_ratio > 0:
#             return features, mask, forward_indexes, backward_indexes
#         return features


# #decoder
# self.pos_emb
# self.task_emb
# project_encoder tokens
# add task and positional embeddings to encoded tokens
# # paper uses task + mask tokens as cross attention queries to everything else, can we do cross mae by just doing mask tokens to everything? then we don't have to separate everything out
# # they also don't remove the task tokens from the contextembeddings so the "cross attention" layer is also doing self attention to the visual tokens of each task basically
# cross_mae from task-specific mask tokens to everything else
