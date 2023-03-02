import math
from abc import ABC
from pathlib import Path

import numpy as np
import torch
from aicsimageio.writers import OmeTiffWriter
from monai.inferers import sliding_window_inference
from monai.networks.blocks import (
    Convolution,
    SubpixelUpsample,
    UnetOutBlock,
    UnetResBlock,
)


class BaseAuxHead(ABC, torch.nn.Module):
    def __init__(
        self, loss, postprocess, model_args=None, calculate_metric=False, inference_args={}
    ):
        super().__init__()
        self.loss = loss
        self.postprocess = postprocess
        self.calculate_metric = calculate_metric
        self.model = self._init_model(model_args)
        self.inference_params = inference_args

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

    def _train_forward(self, x):
        return self.model(x)

    def _inference_forward(self, x):
        # flag for whether to do sliding window
        with torch.no_grad():
            outs = sliding_window_inference(
                inputs=x, predictor=self._train_forward, **self.inference_args
            )
        return outs

    def forward(self, x, stage):
        if stage in ("train", "val"):
            return self._train_forward(x)
        return self._inference_forward(x)

    def run_head(self, backbone_features, batch, stage, save_image, global_step):
        y_hat = self.forward(backbone_features, stage)
        y = batch[self.head_name]
        if stage != "predict":
            loss = self._calculate_loss(y_hat, y)
        if save_image:
            y_hat_out = self._postprocess(y_hat, img_type="predict")
            if stage in ("train", "val"):
                y_out = self._postprocess(y, img_type="input")

            save_name = (
                [f"{global_step}_{self.head_name}.tif"]
                if stage in ("train", "val")
                else batch.get(f"{self.x_key}_meta_dict")
            )
            n_save = len(y_hat_out) if stage in ("test", "predict") else 1
            for i in range(n_save):
                self._save(save_name[i].replace(".tif", "_pred.tif"), y_hat_out[i], stage)
                if stage in ("train", "val"):
                    self._save(save_name[i], y_out[i], stage)
        metric = None
        if self.calculate_metric and stage in ("val", "test"):
            metric = self._calculate_metric(y_hat, y)
        return {"loss": loss, "metric": metric}


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
        self, loss, postprocess, model_args=None, calculate_metric=False, inference_args={}
    ):
        super().__init__(loss, postprocess, model_args, calculate_metric, inference_args)

    def _init_model(self, model_args):
        resolution = model_args.get("resolution", "lr")
        self.resolution = resolution
        spatial_dims = model_args.get("spatial_dims", 3)
        n_convs = model_args.get("n_convs", 1)
        dropout = model_args.get("dropout", 0.0)
        out_channels = model_args["out_channels"]
        final_act = model_args["final_act"]
        in_channels = model_args["in_channels"]

        conv_input_channels = in_channels
        modules = [model_args.get("first_layer", torch.nn.Identity())]
        if resolution == "hr":
            conv_input_channels //= 2**spatial_dims
            self.upsample = SubpixelUpsample(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=conv_input_channels,
            )

        for i in range(n_convs):
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
        return torch.nn.Sequential(*modules)

    def _train_forward(self, x):
        if self.resolution == "hr":
            x = self.upsample(x)
        return self.model(x)
