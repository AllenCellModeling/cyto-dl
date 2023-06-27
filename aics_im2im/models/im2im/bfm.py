import random
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        num_interactive_rounds: int = 11,
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
            "train/seg_loss": MeanMetric(),
            "val/seg_loss": MeanMetric(),
            "test/seg_loss": MeanMetric(),
            "train/qc_loss": MeanMetric(),
            "val/qc_loss": MeanMetric(),
            "test/qc_loss": MeanMetric(),
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

        def compute_prompts(self, preds, target):
            """
            targets: [N_masks, D, H, W]
            preds: [N_masks, D, H, W]
            """

            # we convert logits to probabilities, to compute a collapsed
            # version of the prediction, to be used for error region calculation
            # and for dense prompt calculation
            collapsed_pred = (preds.sigmoid().sum(dim=0) + 1e-9).logit()
            collapsed_target = target.sum(dim=0)

            dense = F.interpolate(collapsed_pred, scale_factor=0.25)

            # after we create the dense prompt, we threshold the collapsed
            # predictions
            collapsed_pred = collapsed_pred.gt(0)

            overseg_region = collapsed_pred - collapsed_target
            underseg_region = collapsed_target - collapsed_pred

            overlap_region = torch.zeros_like(collapsed_pred)
            for ix, pred in enumerate(preds):
                # compute a collapsed target excluding this index (i.e. all other targets)
                collapsed_others = target[:ix] + target[ix + 1 :]

                # compute the region where the model predicted the ix-th object
                # at a location where any other object exists
                overlap_region += pred * collapsed_others

            point_type = random.choices(  # nosec: B311
                (0, 1, 2),
                weights=(overseg_region.sum(), underseg_region.sum(), overlap_region.sum()),
            )

            sample_from = (overseg_region, underseg_region, overlap_region)[point_type].argwhere()
            rand_ix = torch.randint(sample_from.shape[0], (1,))[0]
            point_prompt = sample_from[rand_ix]

            return dense, (point_prompt.unsqueeze(0), point_type)

        def model_step(self, stage, batch, batch_idx):
            images = batch[self.hparams.x_key]
            targets = batch[self.hparams.y_key]
            image_size = images.shape[-3:]
            image_embeddings = self.encoder(images)

            if self.training:
                point_prompts = [None] * images.shape[0]
                dense_prompts = [None] * images.shape[0]

            opt = self.optimizers()

            # choosing what types of prompts are generated at each interactive
            # round. following SAM, there's two rounds where no additional prompts
            # are added. one of these is always at the end and another one
            # is randomly inserted
            rounds = ["points"] * self.hparams.num_interactive_rounds
            rounds[-1] = "nop"

            random_ix = random.randint(1, self.hparams.num_interactive_rounds - 2)  # nosec: B311
            rounds[random_ix] = "nop"

            for round_op in rounds:
                seg_loss = 0
                qc_loss = 0
                for ix, (image, embedding, target, task_index) in enumerate(
                    zip(images, image_embeddings, targets, batch["task_index"])
                ):
                    if round_op == "points":
                        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                            point_prompts[ix], None, dense_prompts[ix], image_size
                        )

                    preds, qc = self.decoder(
                        embedding,
                        self.prompt_encoder.get_dense_pe(),
                        sparse_embeddings,
                        dense_embeddings,
                        task_index,
                        image.shape,
                    )

                    with torch.no_grad():
                        i, j = self.matcher(preds, target)
                        new_point_prompt, dense_prompts[ix] = self.compute_prompts(
                            preds[i], target[j]
                        )
                        point_prompts[ix] = torch.cat((point_prompts[ix], new_point_prompt), dim=0)

                    seg_loss = seg_loss + self.loss(preds[i], target[j])
                    with torch.no_grad():
                        i = (preds[i].gt(0) & target[j]).reshape(images.shape[0], -1).sum(dim=1)
                        u = (preds[i].gt(0) | target[j]).reshape(images.shape[0], -1).sum(dim=1)
                        iou = i / (u + 1e-9)

                    qc_loss = qc_loss + F.mse_loss(qc, iou)

                if stage == "train":
                    opt.zero_grad()

                seg_loss = seg_loss / images.shape[0]
                qc_loss = qc_loss / images.shape[0]

                loss = seg_loss + qc_loss
                if stage == "train":
                    self.manual_backward(loss)
                    opt.step()

            return {"loss": loss, "seg_loss": seg_loss, "qc_loss": qc_loss}, None, None

        def prediction_step(self, batch, batch_idx):
            images = batch[self.hparams.x_key]
            image_size = images.shape[-3:]
            image_embeddings = self.encoder(images)

            point_prompts = batch.get("point_prompts", None)
            dense_prompts = batch.get("dense_prompts", None)

            preds = [None] * images.shape[0]
            for ix, (image, embedding, task_index) in enumerate(
                zip(images, image_embeddings, batch["task_index"])
            ):
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    point_prompts[ix], None, dense_prompts[ix], image_size
                )

                preds[ix] = self.decoder(
                    embedding,
                    self.prompt_encoder.get_dense_pe(),
                    sparse_embeddings,
                    dense_embeddings,
                    task_index,
                    image.shape,
                )

            return preds
