import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from cyto_dl.models.vae.point_cloud_vae import PointCloudVAE
from torch_geometric.nn.models import InnerProductDecoder

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
from torch_geometric.transforms import SamplePoints, KNNGraph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, ResidualUnit, UpSample
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Flatten, Reshape
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss as Loss

from cyto_dl.image.transforms import RotationMask
from cyto_dl.models.vae.base_vae import BaseVAE
from cyto_dl.models.vae.image_vae import ImageVAE
from cyto_dl.utils.rotation import RotationModule

from .image_encoder import ImageEncoder

logger = logging.getLogger("lightning")
logger.propagate = False


class _Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.tensor(scale)

    def forward(self, x):
        return x * self.scale.type_as(x)


class PointCloudVAEAdj(PointCloudVAE):
    def __init__(
        self,
        latent_dim: int,
        latent_dim_inv: int,
        latent_dim_spur: int,
        x_label: str,
        encoder: dict,
        decoder: dict,
        condition_keys: list,
        reconstruction_loss: dict,
        prior: dict,
        get_rotation: bool = False,
        beta: float = 1.0,
        disable_metrics: bool = True,
        image_config: dict = None,
        id_label: Optional[str] = None,
        point_label: Optional[str] = "points",
        occupancy_label: Optional[str] = "points.df",
        embedding_head: Optional[dict] = None,
        embedding_head_loss: Optional[dict] = None,
        embedding_head_weight: Optional[dict] = None,
        condition_encoder: Optional[dict] = None,
        condition_decoder: Optional[dict] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        include_top_loss: Optional[bool] = False,
        topo_lambda: Optional[float] = None,
        topo_num_groups: Optional[int] = None,
        farthest_point: Optional[bool] = True,
        **base_kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            x_label=x_label,
            get_rotation=get_rotation,
            id_label=id_label,
            optimizer=optimizer,
            beta=beta,
            point_label=point_label,
            occupancy_label=occupancy_label,
            encoder=encoder,
            decoder=decoder,
            embedding_head=embedding_head,
            embedding_head_loss=embedding_head_loss,
            embedding_head_weight=embedding_head_weight,
            condition_encoder=condition_encoder,
            condition_decoder=condition_decoder,
            condition_keys=condition_keys,
            reconstruction_loss=reconstruction_loss,
            prior=prior,
            disable_metrics=disable_metrics,
            include_top_loss=include_top_loss,
            topo_lambda=topo_lambda,
            farthest_point=farthest_point,
            topo_num_groups=topo_num_groups,
        )

        # image_vae = ImageVAE(
        #     x_label = 'image',
        #     latent_dim = latent_dim,
        #     spatial_dims = np.asarray(image_config["spatial_dims"]),
        #     in_shape = np.asarray(image_config["in_shape"]),
        #     channels = np.asarray(image_config["channels"]),
        #     strides = np.asarray(image_config["strides"]),
        #     kernel_sizes = np.asarray(image_config["kernel_sizes"]),
        #     out_channels = np.asarray(image_config["out_channels"]),
        #     decoder_channels = np.asarray(image_config["decoder_channels"]),
        #     decoder_strides = np.asarray(image_config["decoder_strides"]),
        #     background_value=0,
        #     act='relu',
        #     norm='batch',
        #     num_res_units=1,
        # )
        # image_vae = ImageVAE(
        #     x_label="image",
        #     latent_dim=256,
        #     spatial_dims=2,
        #     in_shape=[1, 128, 128],
        #     channels=[8, 16, 32, 64, 128, 256, 512],
        #     strides=[1, 1, 2, 2, 2, 2, 2],
        #     kernel_sizes=[3, 3, 3, 3, 3, 3, 3],
        #     decoder_channels=[512, 256, 128, 64, 32, 16],
        #     decoder_strides=[2, 2, 2, 2, 1, 1],
        #     background_value=0,
        #     act="relu",
        #     norm="batch",
        #     num_res_units=1,
        #     prior="identity",
        # )
        image_vae = ImageVAE(
            x_label="image",
            latent_dim=256,
            spatial_dims=2,
            in_shape=[1, 128, 128],
            # channels=[8, 16, 32, 64, 128, 256, 512],
            channels=[4, 4, 4, 4, 4, 4, 4],
            strides=[1, 1, 2, 2, 2, 2, 2],
            kernel_sizes=[3, 3, 3, 3, 3, 3, 3],
            decoder_channels=[4, 4, 4, 4, 4, 4],
            decoder_strides=[2, 2, 2, 2, 1, 1],
            background_value=0,
            act="relu",
            norm="batch",
            num_res_units=1,
            group=None,
            prior="identity",
        )
        self.encoder_adj = image_vae.encoder["embedding"]
        self.decoder_adj = image_vae.decoder["image"]
        self.adj_loss = torch.nn.MSELoss(reduction=None)
        self.condition_keys = None

    def calculate_rcl(self, batch, xhat, input_key, target_key=None):
        if not target_key:
            target_key = input_key
        rcl_per_input_dimension = self.reconstruction_loss[input_key](
            batch[target_key], xhat[input_key]
        )

        if torch.isnan(rcl_per_input_dimension).any():
            mask = torch.isnan(rcl_per_input_dimension).bool()
            rcl_per_input_dimension = rcl_per_input_dimension[~mask]

        if (self.mask_keys is not None) and (self.target_mask_keys is not None):
            this_mask = batch["target_mask"].type_as(rcl_per_input_dimension).byte()
            rcl_per_input_dimension = rcl_per_input_dimension * ~this_mask.bool()

        return rcl_per_input_dimension

    def decode(self, z_parts, return_canonical=False, batch=None):
        if hasattr(self.encoder[self.hparams.x_label], "generate_grid_feats"):
            if self.encoder[self.hparams.x_label].generate_grid_feats:
                base_xhat = self.decoder[self.hparams.x_label](
                    batch[self.point_label], z_parts["grid_feats"]
                )
            else:
                base_xhat = self.decoder[self.hparams.x_label](
                    z_parts[self.hparams.x_label]
                )
        else:
            base_xhat = self.decoder[self.hparams.x_label](
                z_parts[self.hparams.x_label]
            )

        if self.get_rotation:
            rotation = z_parts["rotation"]
            xhat = torch.einsum("bij,bjk->bik", base_xhat[:, :, :3], rotation)
            if xhat.shape[-1] != base_xhat.shape[-1]:
                xhat = torch.cat([xhat, base_xhat[:, :, -1:]], dim=-1)
        else:
            xhat = base_xhat
        adj_pred = self.decoder_adj(z_parts[self.hparams.x_label])
        if return_canonical:
            return {
                self.hparams.x_label: xhat,
                "canonical": base_xhat,
                "adj": adj_pred,
            }

        return {self.hparams.x_label: xhat, "adj": adj_pred}

    def encode(self, batch, **kwargs):
        ret_dict = {}
        for part, encoder in self.encoder.items():
            this_batch_part = batch[part]
            this_ret = encoder(
                this_batch_part,
                **{k: v for k, v in kwargs.items() if k in self.encoder_args[part]},
            )

            if isinstance(this_ret, dict):  # deal with multiple outputs for an encoder
                for key in this_ret.keys():
                    ret_dict[key] = this_ret[key]
            else:
                ret_dict[part] = this_ret
        ret_dict["adj"] = self.encoder_adj(batch["adj"])
        return ret_dict

    def encoder_compose_function(self, z_parts, batch):
        batch_size = z_parts[self.hparams.x_label].shape[0]

        pcloud_embed = z_parts[self.hparams.x_label]
        adj_embed = z_parts["adj"]
        cat_embed = torch.cat([pcloud_embed, adj_embed], dim=1)
        # shared encoder
        z_parts[self.hparams.x_label] = self.condition_encoder[self.hparams.x_label](
            cat_embed
        )
        if self.embedding_head:
            for key in self.embedding_head.keys():
                z_parts[key] = self.embedding_head[key](z_parts[self.hparams.x_label])

        return z_parts
