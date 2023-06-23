from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from aics_imi2m.nn.bfm import (
    HungarianMatcher,
    MaskDecoder,
    PromptEncoder,
    TwoWayTransformer,
)
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from torch.nn.modules.loss import _Loss as Loss
from torchmetrics import MeanMetric, MinMetric

from aics_im2im.models.base_model import BaseModel


class BFM(BaseModel):
    def __init__(
        self,
        *,
        x_key: str,
        y_key: str,
        encoder_ckpt,
        num_multimask_outputs: int = 5,
        num_hungarian_matching_points: int = 1000,
        image_embedding_size: Tuple[int, int, int],
        # SAM default arguments
        num_interactive_rounds: int = 8,
        activation: Type[nn.Module] = nn.GELU,
        transformer_dim: int = 256,
        qc_head_depth: int = 3,
        qc_head_hidden_dim: int = 256,
        transformer_depth: int = 2,
        transformer_mlp_dim: int = 2048,
        transformer_num_heads: int = 8,
        loss: Loss = DiceFocalLoss(),  # TODO: pick default parameters
        **base_kwargs,
    ):
        """
        Parameters
        ----------
        backbone: nn.Module
            backbone network, parameters are shared between task heads
        task_heads: Dict
            task-specific heads
        x_key: str
            key of input image in batch
        save_dir="./"
            directory to save images during training and validation
        save_images_every_n_epochs=1
            Frequency to save out images during training
        inference_args: Dict = {}
            Arguments passed to monai's [sliding window inferer](https://docs.monai.io/en/stable/inferers.html#sliding-window-inference)
        inference_heads: Union[List, None] = None
            Optional list of heads to run during inference. Defaults to running all heads.
        **base_kwargs:
            Additional arguments passed to BaseModel
        """

        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
        }

        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.encoder = ""  # TODO: load encoder from ckpt
        self.decoder = MaskDecoder(
            num_multimask_outputs=num_multimask_outputs,
            transformer=TwoWayTransformer(
                depth=transformer_depth,
                embedding_dim=transformer_dim,
                mlp_dim=transformer_mlp_dim,
                num_heads=transformer_num_heads,
            ),
            transformer_dim=transformer_dim,
            qc_head_depth=qc_head_depth,
            qc_head_hidden_dim=qc_head_hidden_dim,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=transformer_dim,
            image_embedding_size=image_embedding_size,
            mask_in_chans=16,
        )

        self.matcher = HungarianMatcher(1, 1, num_hungarian_matching_points)
        self.loss = loss

        def forward(self, batch, sparse_prompts=None, dense_prompts=None):
            # TODO: this was copied from segment_anything. adjust
            """
            input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
            image_embeddings = self.image_encoder(input_images)

            outputs = []
            for image_record, curr_embedding in zip(batched_input, image_embeddings):
                if "point_coords" in image_record:
                    points = (image_record["point_coords"], image_record["point_labels"])
                else:
                    points = None
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )
                low_res_masks, qc_predictions = self.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                )
                masks = self.postprocess_masks(
                    low_res_masks,
                    input_size=image_record["image"].shape[-2:],
                    original_size=image_record["original_size"],
                )
                masks = masks > self.mask_threshold
                outputs.append(
                    {
                        "masks": masks,
                        "qc_predictions": qc_predictions,
                        "low_res_logits": low_res_masks,
                    }
                )
            return outputs
            """
            pass

        def compute_prompts(self, target, preds):
            """
            targets: [N_masks, D, H, W]
            preds: [N_masks, D, H, W]
            """
            return "point", "dense"

        def training_step(self, batch):
            # TODO:
            # training:
            # - do n rounds of predicting, computing loss and backprop, and feeding previous
            #   prediction as dense prompt, and automatically generate simulated sparse prompts

            images = batch[self.hparams.x_key]
            targets = batch[self.hparams.y_key]
            image_size = images.shape[-3:]
            image_embeddings = self.encoder(images)

            if self.training:
                # TODO: make this something other than a list?
                point_prompts = [None] * images.shape[0]
                dense_prompts = [None] * images.shape[0]
            else:
                raise NotImplementedError

            opt = self.optimizers()

            for _ in range(self.hparams.num_interactive_rounds):
                loss = 0
                for ix, (image, embedding, target) in enumerate(
                    zip(images, image_embeddings, targets)
                ):
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        point_prompts[ix], None, dense_prompts[ix], image_size
                    )

                    preds = self.decoder(
                        embedding,
                        self.prompt_encoder.get_dense_pe(),
                        sparse_embeddings,
                        dense_embeddings,
                    )

                    point_prompts[ix], dense_prompts[ix] = self.compute_prompts(target, preds)

                    with torch.no_grad():
                        i, j = self.matcher(preds, target)

                    loss = loss + self.loss(preds[i], target[j])

                opt.zero_grad()
                loss = loss / images.shape[0]
                self.manual_backward(loss)
                opt.step()

            return {"loss": loss}, None, None
