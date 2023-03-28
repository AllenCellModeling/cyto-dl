import math
from abc import ABC
from pathlib import Path

import numpy as np
import torch
from aicsimageio.writers import OmeTiffWriter
from monai.networks.blocks import Convolution, UnetOutBlock, UnetResBlock, UpSample

from aics_im2im.models.im2im.utils.postprocessing import detach
from aics_im2im.nn.losses import pix2pix_hd


class BaseAuxHead(ABC, torch.nn.Module):
    def __init__(
        self,
        loss,
        postprocess={"input": detach, "prediction": detach},
        model_args=None,
        calculate_metric=False,
        save_raw=False,
    ):
        super().__init__()
        self.loss = loss
        self.postprocess = postprocess
        self.calculate_metric = calculate_metric
        self.model = self._init_model(model_args)
        self.save_raw = save_raw

    def update_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def _init_model(self, model_args):
        return torch.nn.Sequential(torch.nn.Identity())

    def _calculate_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def _postprocess(self, img, img_type):
        return [self.postprocess[img_type](img[i]) for i in range(img.shape[0])]

    def _save(self, fn, img, stage):
        OmeTiffWriter().save(
            uri=Path(self.save_dir) / f"{stage}_images" / fn,
            data=img.squeeze().astype(float),
            dims_order="STCZYX"[-len(img.shape)],
        )

    def _calculate_metric(self, y_hat, y):
        raise NotImplementedError

    def save_image(self, y_hat, batch, stage, global_step):
        y_hat_out = self._postprocess(y_hat, img_type="prediction")
        y_out, raw_out = None, None
        if stage in ("train", "val"):
            y_out = self._postprocess(batch[self.head_name], img_type="input")
            if self.save_raw:
                raw_out = self._postprocess(batch[self.x_key], img_type="input")
        try:
            metadata_filenames = batch[f"{self.x_key}_meta_dict"]["filename_or_obj"]
            metadata_filenames = [
                f"{Path(fn).stem}_{self.head_name}.tif" for fn in metadata_filenames
            ]
        except KeyError:
            raise ValueError(
                f"Please ensure your batches contain key `{self.x_key}_meta_dict['filename_or_obj']`"
            )
        save_name = (
            [f"{global_step}_{self.head_name}.tif"]
            if stage in ("train", "val")
            else metadata_filenames
        )
        n_save = len(y_hat_out) if stage in ("test", "predict") else 1
        for i in range(n_save):
            self._save(save_name[i].replace(".tif", "_pred.tif"), y_hat_out[i], stage)
            if stage in ("train", "val"):
                self._save(save_name[i], y_out[i], stage)
                if self.save_raw:
                    self._save(save_name[i].replace(".tif", "_raw.tif"), raw_out[i], stage)

        return y_hat_out, y_out

    def forward(self, x):
        return self.model(x)

    def run_head(
        self,
        backbone_features,
        batch,
        stage,
        save_image,
        global_step,
        run_forward=True,
        y_hat=None,
    ):
        if run_forward:
            y_hat = self.forward(backbone_features)
        if y_hat is None:
            raise ValueError(
                "y_hat must be provided, either by passing it in or setting `run_forward=True`"
            )
        loss = None
        if stage != "predict":
            loss = self._calculate_loss(y_hat, batch[self.head_name])

        y_hat_out, y_out = None, None
        if save_image:
            y_hat_out, y_out = self.save_image(y_hat, batch, stage, global_step)

        metric = None
        if self.calculate_metric and stage in ("val", "test"):
            metric = self._calculate_metric(y_hat, batch[self.head_name])
        return {"loss": loss, "metric": metric, "y_hat_out": y_hat_out, "y_out": y_out}


class ConvProjectionLayer(torch.nn.Module):
    def __init__(self, dim, pool_size, in_channels, out_channels):
        super().__init__()
        self.dim = dim
        n_downs = math.floor(np.log2(pool_size))
        modules = []
        for _ in range(n_downs):
            modules.append(
                Convolution(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=[2, 1, 1],
                    strides=[2, 1, 1],
                    padding=[0, 0, 0],
                )
            )
        remainder = pool_size - 2**n_downs
        if remainder != 0:
            modules.append(
                Convolution(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=[remainder, 1, 1],
                    strides=[remainder, 1, 1],
                    padding=[0, 0, 0],
                )
            )
        self.model = torch.nn.Sequential(*modules)

    def __call__(self, x):
        return self.model(x).squeeze(self.dim)


class AuxHead(BaseAuxHead):
    def __init__(
        self,
        loss,
        postprocess={"input": detach, "prediction": detach},
        model_args=None,
        calculate_metric=False,
        save_raw=False,
    ):
        super().__init__(loss, postprocess, model_args, calculate_metric, save_raw)

    def _init_model(self, model_args):
        resolution = model_args.get("resolution", "lr")
        self.resolution = resolution
        spatial_dims = model_args.get("spatial_dims", 3)
        n_convs = model_args.get("n_convs", 1)
        dropout = model_args.get("dropout", 0.0)
        out_channels = model_args["out_channels"]
        final_act = model_args["final_act"]
        in_channels = model_args["in_channels"]
        upsample_method = model_args.get("upsample_method", "subpixel")

        conv_input_channels = in_channels
        modules = [model_args.get("first_layer", torch.nn.Identity())]
        upsample = torch.nn.Identity()
        if resolution == "hr":
            if upsample_method == "subpixel":
                conv_input_channels //= 2**spatial_dims
            upsample_ratio = model_args.get("upsample_ratio", [2] * self.spatial_dims)
            assert len(upsample_ratio) == spatial_dims
            upsample = UpSample(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=conv_input_channels,
                scale_factor=upsample_ratio,
                mode=upsample_method,
            )
        for _ in range(n_convs):
            in_channels = conv_input_channels
            modules.append(
                UnetResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=conv_input_channels,
                    stride=1,
                    kernel_size=3,
                    norm_name="INSTANCE",
                    dropout=dropout,
                )
            )
        modules.extend(
            (
                UnetOutBlock(
                    spatial_dims=spatial_dims,
                    in_channels=conv_input_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                ),
                final_act,
            )
        )
        model = torch.nn.ModuleDict({"upsample": upsample, "model": torch.nn.Sequential(*modules)})
        return model

    def forward(self, x):
        if self.resolution == "hr":
            x = self.model["upsample"](x)
        return self.model["model"](x)


class GANHead(BaseAuxHead):
    """GAN Task head."""

    def __init__(
        self,
        gan_loss=pix2pix_hd(scales=1),
        reconstruction_loss=torch.nn.MSELoss(),
        reconstruction_loss_weight=100,
        postprocess={"input": detach, "prediction": detach},
        model_args=None,
        calculate_metric=False,
        save_raw=False,
    ):
        """
        Parameters
        ----------
        gan_loss=pix2pix_hd(scales=1)
            Loss for optimizing GAN
        reconstruction_loss=torch.nn.MSELoss()
            Loss for optimizing generator's image reconstructions
        reconstruction_loss_weight=100
            Weighting of reconstruction loss
        postprocess={"input": detach, "prediction": detach}
            Postprocessing for `input` and `predictions` of head
        model_args=None
            Arguments to initialize model. This is not used by the GAN head
        calculate_metric=False
            Whether to calculate a metric during training. Not used by GAN head.
        save_raw=False
            Whether to save out example input images during training
        """
        super().__init__(None, postprocess, model_args, calculate_metric, save_raw)
        self.gan_loss = gan_loss
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_loss_weight = reconstruction_loss_weight

    def _init_model(self, model_args):
        return torch.nn.Sequential(torch.nn.Tanh())

    def _calculate_loss(self, y_hat, batch, discriminator):
        # extract intermediate activations from discriminator for real and predicted images
        features_discriminator = discriminator(
            batch[self.x_key], batch[self.head_name], y_hat.detach()
        )
        loss_D = self.gan_loss(features_discriminator, "discriminator")

        # passability of generated images
        features_generator = discriminator(batch[self.x_key], batch[self.head_name], y_hat)
        loss_G = self.gan_loss(features_generator, "generator")
        # image reconstruction quality
        loss_reconstruction = self.reconstruction_loss(batch[self.head_name], y_hat)
        return loss_D, loss_G + loss_reconstruction * self.reconstruction_loss_weight

    def run_head(
        self,
        backbone_features,
        batch,
        stage,
        save_image,
        global_step,
        discriminator=None,
        run_forward=True,
        y_hat=None,
    ):
        if run_forward:
            y_hat = self.forward(backbone_features)
        if y_hat is None:
            raise ValueError(
                "y_hat must be provided, either by passing it in or setting `run_forward=True`"
            )
        loss_D, loss_G = None, None
        if stage != "predict":
            if discriminator is None:
                raise ValueError(
                    "Discriminator must be specified for train, test, and validation steps."
                )
            loss_D, loss_G = self._calculate_loss(y_hat, batch, discriminator)

        y_hat_out, y_out = None, None
        if save_image:
            y_hat_out, y_out = self.save_image(y_hat, batch, stage, global_step)

        metric = None
        if self.calculate_metric and stage in ("val", "test"):
            metric = self._calculate_metric(y_hat, batch[self.head_name])

        return {
            "loss_D": loss_D,
            "loss_G": loss_G,
            "metric": metric,
            "y_hat_out": y_hat_out,
            "y_out": y_out,
        }
