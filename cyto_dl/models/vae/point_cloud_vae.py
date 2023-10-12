import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from cyto_dl.models.vae.base_vae import BaseVAE
from cyto_dl.models.vae.priors import IdentityPrior, IsotropicGaussianPrior
from cyto_dl.nn.losses import ChamferLoss
from cyto_dl.nn.point_cloud import DGCNN, FoldingNet
# from topologylayer.nn import AlphaLayer, BarcodePolyFeature
from cyto_dl.nn.losses import TopoLoss

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class PointCloudVAE(BaseVAE):
    def __init__(
        self,
        latent_dim: int,
        x_label: str,
        encoder: dict,
        decoder: dict,
        reconstruction_loss: dict,
        prior: dict,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        get_rotation=False,
        beta: float = 1.0,
        num_points: Optional[int] = None,
        hidden_dim: Optional[int] = 64,
        hidden_conv2d_channels: Optional[list] = [64, 64, 64, 64],
        hidden_conv1d_channels: Optional[list] = [512, 20],
        hidden_decoder_dim: Optional[int] = 512,
        k: Optional[int] = 20,
        mode: Optional[str] = "scalar",
        include_cross: Optional[bool] = False,
        include_coords: Optional[bool] = True,
        id_label: Optional[str] = None,
        embedding_prior: Optional[str] = "identity",
        decoder_type: Optional[str] = "foldingnet",
        loss_type: Optional[str] = "chamfer",
        eps: Optional[float] = 1e-6,
        shape: Optional[str] = "sphere",
        num_coords: Optional[int] = 3,
        std: Optional[float] = 0.3,
        sphere_path: Optional[str] = None,
        gaussian_path: Optional[str] = None,
        symmetry_breaking_axis: Optional[Union[str, int]] = None,
        scalar_inds: Optional[int] = None,
        generate_grid_feats: Optional[bool] = False,
        padding: Optional[float] = 0.1,
        reso_plane: Optional[int] = 64,
        plane_type: Optional[list] = ["xz", "xy", "yz"],
        scatter_type: Optional[str] = "max",
        point_label: Optional[str] = "points",
        occupancy_label: Optional[str] = "points.df",
        embedding_head: Optional[dict] = None,
        embedding_head_loss: Optional[dict] = None,
        embedding_head_weight: Optional[dict] = None,
        basal_head: Optional[dict] = None,
        basal_head_loss: Optional[dict] = None,
        basal_head_weight: Optional[dict] = None,
        condition_encoder: Optional[dict] = None,
        condition_decoder: Optional[dict] = None,
        condition_keys: Optional[list] = None,
        mask_keys: Optional[list] = None,
        masking_ratio: Optional[float] = None,
        disable_metrics: Optional[bool] = False,
        metric_keys: Optional[list] = None,
        include_top_loss: Optional[bool] = False,
        inference_mask_dict: Optional[dict] = None,
        parse: Optional[bool] = False,
        **base_kwargs,
    ):
        self.get_rotation = get_rotation
        self.symmetry_breaking_axis = symmetry_breaking_axis
        self.metric_keys = metric_keys
        self.scalar_inds = scalar_inds
        self.decoder_type = decoder_type
        self.generate_grid_feats = generate_grid_feats
        self.occupancy_label = occupancy_label
        self.point_label = point_label
        self.condition_keys = condition_keys
        self.embedding_head = embedding_head
        self.embedding_head_loss = embedding_head_loss
        self.embedding_head_weight = embedding_head_weight
        self.basal_head = basal_head
        self.basal_head_loss = basal_head_loss
        self.basal_head_weight = basal_head_weight
        self.disable_metrics = disable_metrics
        self.include_top_loss = include_top_loss
        self.parse = parse
        self.mask_keys = mask_keys
        self.masking_ratio = masking_ratio

        if embedding_prior == "gaussian":
            self.encoder_out_size = 2 * latent_dim
        else:
            self.encoder_out_size = latent_dim

        if encoder is None:
            encoder = DGCNN(
                num_features=self.encoder_out_size,
                hidden_dim=hidden_dim,
                hidden_conv2d_channels=hidden_conv2d_channels,
                hidden_conv1d_channels=hidden_conv1d_channels,
                k=k,
                mode=mode,
                scalar_inds=scalar_inds,
                include_cross=include_cross,
                include_coords=include_coords,
                symmetry_breaking_axis=symmetry_breaking_axis,
                generate_grid_feats=generate_grid_feats,
                padding=padding,
                reso_plane=reso_plane,
                plane_type=plane_type,
                scatter_type=scatter_type,
            )
            encoder = {x_label: encoder}

        if decoder is None:
            if decoder_type == "foldingnet":
                decoder = FoldingNet(
                    latent_dim,
                    num_points,
                    hidden_decoder_dim,
                    std,
                    shape,
                    sphere_path,
                    gaussian_path,
                    num_coords,
                )
            else:
                raise ValueError(f"Key`{decoder_type}` is not implemented")
            decoder = {x_label: decoder}

        if reconstruction_loss is None:
            if loss_type == "chamfer":
                reconstruction_loss = {x_label: ChamferLoss()}
            elif loss_type == "L1":
                reconstruction_loss = {x_label: torch.nn.L1Loss(reduction="mean")}

        if prior is None:
            prior = {
                "embedding": (
                    IsotropicGaussianPrior(dimensionality=latent_dim)
                    if embedding_prior == "gaussian"
                    else IdentityPrior(dimensionality=latent_dim)
                ),
            }

        if self.get_rotation:
            prior["rotation"] = IdentityPrior(dimensionality=1)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            x_label=x_label,
            id_label=id_label,
            beta=beta,
            reconstruction_loss=reconstruction_loss,
            optimizer=optimizer,
            prior=prior,
            disable_metrics=disable_metrics,
            metric_keys=metric_keys,
        )

        self.condition_encoder = nn.ModuleDict(condition_encoder)
        self.condition_decoder = nn.ModuleDict(condition_decoder)
        self.embedding_head = nn.ModuleDict(embedding_head)
        self.embedding_head_loss = nn.ModuleDict(embedding_head_loss)
        self.basal_head = nn.ModuleDict(basal_head)
        self.basal_head_loss = nn.ModuleDict(basal_head_loss)
        self.inference_mask_dict = inference_mask_dict
        if self.include_top_loss:
            # self.top_layer = nn.ModuleDict({x_label: AlphaLayer(maxdim=1)})
            # self.top_loss = nn.ModuleDict({x_label: BarcodePolyFeature(1,2,0)})
            self.top_loss = nn.ModuleDict({x_label: TopoLoss()})

    def encode(self, batch, **kwargs):
        ret_dict = {}
        for part, encoder in self.encoder.items():
            this_batch_part = batch[part]
            if self.mask_keys:
                # if mask, then mask this batch part
                if f"{part}" in self.mask_keys:
                    # mask is 1 for batch elements to mask, 0 otherwise
                    this_mask = batch[f"{part}_mask"].byte().repeat(1, this_batch_part.shape[-1])
                    # multiply inverse mask with batch part, so every mask element of 1 is set to 0
                    this_batch_part = this_batch_part * ~this_mask.bool()
            this_ret = encoder(
                this_batch_part,
                **{k: v for k, v in kwargs.items() if k in self.encoder_args[part]},
            )
            if isinstance(this_ret, dict):  # deal with multiple outputs for an encoder
                for key in this_ret.keys():
                    ret_dict[key] = this_ret[key]
            else:
                ret_dict[part] = this_ret
        return ret_dict

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

        if return_canonical:
            return {self.hparams.x_label: xhat, "canonical": base_xhat}

        return {self.hparams.x_label: xhat}

    def encoder_compose_function(self, z_parts):
        batch_size = z_parts[self.hparams.x_label].shape[0]
        if self.basal_head:
            z_parts[self.hparams.x_label + "_basal"] = z_parts[self.hparams.x_label]
            for key in self.basal_head.keys():
                z_parts[key] = self.basal_head[key](
                    z_parts[self.hparams.x_label + "_basal"]
                )

        if self.condition_keys:
            for j, key in enumerate([self.hparams.x_label] + self.condition_keys):
                this_z_parts = z_parts[key]
                if len(this_z_parts.shape) == 3:
                    this_z_parts = torch.squeeze(z_parts[key], dim=(-1))
                    z_parts[key] = this_z_parts
                    # this_z_parts = this_z_parts.argmax(dim=1)
                this_z_parts = this_z_parts.view(batch_size, -1)
                if j == 0:
                    cond_feats = this_z_parts
                else:
                    cond_feats = torch.cat((cond_feats, this_z_parts), dim=1)
            # shared encoder
            z_parts[self.hparams.x_label] = self.condition_encoder[
                self.hparams.x_label
            ](cond_feats)
        if self.embedding_head:
            for key in self.embedding_head.keys():
                z_parts[key] = self.embedding_head[key](z_parts[self.hparams.x_label])

        return z_parts

    def decoder_compose_function(self, z_parts, batch):
        if (self.condition_keys is not None) & (len(self.condition_decoder.keys()) != 0):
            for j, key in enumerate(self.condition_keys):
                this_batch_part = batch[key]
                if self.mask_keys:
                    # if mask, then mask this batch part
                    if f"{key}" in self.mask_keys:
                        this_mask = batch[f"{key}_mask"].byte().repeat(1, this_batch_part.shape[-1])
                        # multiply inverse mask with batch part, so every mask element of 1 is set to 0
                        this_batch_part = this_batch_part * ~this_mask.bool()

                if j == 0:
                    cond_inputs = this_batch_part
                    # cond_inputs = torch.squeeze(batch[key], dim=(-1))
                else:
                    cond_inputs = torch.cat((cond_inputs, this_batch_part), dim=1)
                cond_feats = torch.cat(
                    (cond_inputs, z_parts[self.hparams.x_label]), dim=1
                )
            # shared decoder
            z_parts[self.hparams.x_label] = self.condition_decoder[
                self.hparams.x_label
            ](cond_feats)
        return z_parts

    def calculate_rcl(self, batch, xhat, input_key, target_key=None):
        if not target_key:
            target_key = input_key
        rcl_per_input_dimension = self.reconstruction_loss[input_key](
            batch[target_key], xhat[input_key]
        )

        if self.mask_keys is not None:
            if f"{input_key}_mask" in batch.keys():
                C_mask = batch[f"{input_key}_mask"].byte().squeeze()
                rcl_per_input_dimension[C_mask] = 0
        return rcl_per_input_dimension

    def calculate_rcl_dict(self, batch, xhat, z):
        rcl_per_input_dimension = {}
        rcl_reduced = {}
        for key in xhat.keys():
            rcl_per_input_dimension[key] = self.calculate_rcl(
                batch, xhat, key, self.occupancy_label
            )
            if len(rcl_per_input_dimension[key].shape) > 0:
                rcl = (
                    rcl_per_input_dimension[key]
                    # flatten
                    .view(rcl_per_input_dimension[key].shape[0], -1)
                    # and sum across each batch element's dimensions
                    .sum(dim=1)
                )

                rcl_reduced[key] = rcl.mean()
            else:
                rcl_reduced[key] = rcl_per_input_dimension[key]

        if self.embedding_head_loss:
            for key in self.embedding_head_loss.keys():
                rcl_reduced[key] = self.embedding_head_weight[
                    key
                ] * self.embedding_head_loss[key](z[key], x[key])

        if self.basal_head_loss:
            for key in self.basal_head_loss.keys():
                rcl_reduced[key] = self.basal_head_weight[key] * self.basal_head_loss[
                    key
                ](z[key], x[key])

        # if (self.include_top_loss) & (self.current_epoch > 10):
        if (self.include_top_loss):
            for key in self.top_loss.keys():
                # top_losses = []
                # for i in range(xhat[key].shape[0]):
                #     top_losses.append(self.top_loss[key](self.top_layer[
                #         key
                #     ](xhat[key][i])))
                # rcl_reduced['top'] = torch.stack(top_losses).mean()
                rcl_reduced['top'] = self.top_loss[key](batch[key], xhat[key])

        return rcl_reduced

    def parse_batch(self, batch, inference):
        if self.parse:
            for key in batch.keys():
                if len(batch[key].shape) == 1:
                    batch[key] = batch[key].unsqueeze(dim=-1)
            if self.mask_keys is not None:
                for key in self.mask_keys:
                    C = batch[key]
                    if self.inference_mask_dict:
                        this_mask = self.inference_mask_dict[key]
                    else:
                        this_mask = 0
                    # get random mask and save to batch
                    C_mask = torch.zeros(C.shape[0]).bernoulli_(this_mask).byte()
                    batch[f"{key}_mask"] = C_mask.unsqueeze(dim=-1).float().type_as(C)

        return batch

    def forward(self, batch, decode=False, inference=True, return_params=False):
        is_inference = inference or not self.training

        batch = self.parse_batch(batch, inference=inference)
        z_params = self.encode(batch, get_rotation=self.get_rotation)
        z_params = self.encoder_compose_function(z_params)

        z = self.sample_z(z_params, inference=inference)

        z = self.decoder_compose_function(z, batch)

        if not decode:
            return z

        if hasattr(self.encoder[self.hparams.x_label], "generate_grid_feats"):
            if self.encoder[self.hparams.x_label].generate_grid_feats:
                xhat = self.decode(z, batch=batch)
            else:
                xhat = self.decode(z)
        else:
            xhat = self.decode(z)

        if return_params:
            return xhat, z, z_params

        return xhat, z

