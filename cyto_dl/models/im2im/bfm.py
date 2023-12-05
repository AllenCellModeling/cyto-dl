import random
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aicsimageio.writers import OmeTiffWriter
from einops import rearrange
from monai.losses import DiceFocalLoss
from torch.nn.modules.loss import _Loss as Loss
from torchmetrics import MeanMetric

from cyto_dl.models.base_model import BaseModel
from cyto_dl.nn.bfm import (
    HungarianMatcher,
    MaskDecoder,
    PromptEncoder,
    TwoWayTransformer,
)


class BFM(BaseModel):
    def __init__(
        self,
        *,
        x_key: str,
        y_key: str,
        encoder,
        save_dir,
        num_multimask_outputs: int = 5,
        num_hungarian_matching_points: int = 1000,
        image_embedding_size: Tuple[int, int, int],
        # SAM default arguments
        num_interactive_rounds: int = 11,
        activation: Type[nn.Module] = nn.GELU,
        transformer_dim: int = 256,
        qc_head_depth: int = 3,
        transformer_depth: int = 2,
        transformer_mlp_dim: int = 2048,
        transformer_num_heads: int = 8,
        loss: Loss = DiceFocalLoss(sigmoid=True, lambda_focal=20),
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
            "train/loss/loss": MeanMetric(),
            "val/loss/loss": MeanMetric(),
            "test/loss/loss": MeanMetric(),
            "train/loss/seg_loss": MeanMetric(),
            "val/loss/seg_loss": MeanMetric(),
            "test/loss/seg_loss": MeanMetric(),
            "train/loss/qc_loss": MeanMetric(),
            "val/loss/qc_loss": MeanMetric(),
            "test/loss/qc_loss": MeanMetric(),
        }

        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.encoder = encoder

        for param in self.encoder.parameters():
            param.requires_grad = False

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
            qc_head_hidden_dim=transformer_dim,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=transformer_dim,
            image_embedding_size=image_embedding_size,
            mask_in_chans=16,
        )
        # self.adaptor= torch.nn.LazyConv3d(256, kernel_size=1, bias=False)
        self.matcher = HungarianMatcher(1, 1, num_hungarian_matching_points)
        self.loss = loss

        self.automatic_optimization = False
        self.image_embedding_size = np.asarray(image_embedding_size)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "frequency": 1,
                },
            }
        return optimizer

    def compute_prompts(self, preds, target):
        """
        targets: [N_masks, D, H, W]
        preds: [N_masks, D, H, W]
        """

        # we convert logits to probabilities, to compute a collapsed
        # version of the prediction, to be used for error region calculation
        # and for dense prompt calculation
        eps = 1e-6
        collapsed_pred = (preds.sigmoid().sum(dim=0)).clip(eps, 1 - eps).logit()
        collapsed_target = target.sum(dim=0).gt(0)

        # unsqueeze to BCZYX for interpolation, then back to ZYX
        # HERE WE NEED TO INTERPOLATE TO FIXED SIZE [8, 128, 128] so that
        # 4x conv downsampled embeddings match image embedding size (2, 32, 32)
        dense = (
            F.interpolate(
                collapsed_pred.unsqueeze(0).unsqueeze(0), size=list(self.image_embedding_size * 4)
            )
            .squeeze(0)
            .squeeze(0)
        )

        # after we create the dense prompt, we threshold the collapsed
        # predictions
        collapsed_pred = collapsed_pred.gt(0)
        # TODO select points based on distance https://arxiv.org/pdf/2307.01187.pdf
        overseg_region = torch.logical_and(collapsed_pred, ~collapsed_target)
        underseg_region = torch.logical_and(~collapsed_pred, collapsed_target)

        overlap_region = torch.zeros_like(collapsed_pred)
        for ix, pred in enumerate(preds):
            # compute a collapsed target excluding this index (i.e. all other targets)
            collapsed_others = torch.cat((target[:ix], target[ix + 1 :]), dim=0).sum(dim=0).gt(0)

            # compute the region where the model predicted the ix-th object
            # at a location where any other object exists
            overlap_region += torch.logical_and(pred.gt(0), collapsed_others)
        point_type = random.choices(  # nosec: B311
            (0, 1, 2),
            weights=(overseg_region.sum(), underseg_region.sum(), overlap_region.sum()),
        )[0]

        sample_from = (overseg_region, underseg_region, overlap_region)[point_type].argwhere()
        if sample_from.shape[0] == 0:
            rand_ix = torch.randint(sample_from.shape[0], (1,))[0]
            point_prompt = sample_from[rand_ix]
        else:
            return dense, (None, None)

        return dense, (point_prompt.unsqueeze(0).unsqueeze(0), torch.tensor([[point_type]]))

    def save_images(self, pred, target):
        OmeTiffWriter.save(
            pred.astype(float), Path(self.hparams.save_dir) / f"{self.current_epoch}_pred.tif"
        )
        OmeTiffWriter.save(
            (target > 0).astype(np.uint8),
            Path(self.hparams.save_dir) / f"{self.current_epoch}_target.tif",
        )

    def model_step(self, stage, batch, batch_idx):
        if self.current_epoch > 200:
            for param in self.encoder.parameters():
                param.requires_grad = True
        # TODO: add task_index to dataloader based on target type
        batch["task_index"] = [0]
        images = batch[self.hparams.x_key].as_tensor()
        targets = batch[self.hparams.y_key].as_tensor()
        image_size = images.shape[-3:]

        # tell vit to not mask the input image
        # features are (token batch channel) order
        image_embeddings, patch_size = self.encoder(images, do_mask=False)
        # if image_embeddings[1] == self.hparams.transformer_dim:
        #     self.adaptor = torch.nn.Identity()

        # image_embeddings = self.adaptor(image_embeddings)

        point_prompts = [None] * images.shape[0]
        point_labels = [None] * images.shape[0]
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

        for round_num, round_op in enumerate(rounds):
            seg_loss = 0
            qc_loss = 0
            for ix, (image, embedding, target, task_index) in enumerate(
                zip(images, image_embeddings, targets, batch["task_index"])
            ):
                if round_op == "points":
                    points = (
                        (point_prompts[ix], point_labels[ix])
                        if point_prompts[ix] is not None
                        else None
                    )
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points, None, dense_prompts[ix], image_size
                    )
                else:
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        None, None, None, image_size
                    )
                preds, qc = self.decoder(
                    embedding,
                    self.prompt_encoder.get_dense_pe(),
                    sparse_embeddings,
                    dense_embeddings,
                    task_index,
                    image.shape[-3:],
                )

                # note that we can have more target masks than predicted masks. In this case, match randomly keep number of predicted masks from target
                if target.max() > 4:
                    # shuffle target labels and only keep top 4
                    tensor_values = target.unique()
                    tensor_values = tensor_values[tensor_values > 0]
                    tensor_values = tensor_values[torch.randperm(tensor_values.shape[0])][
                        : self.hparams.num_multimask_outputs - 1
                    ]  # -1 for background

                    current_max = torch.max(target)
                    for i, v in enumerate(tensor_values, start=1):
                        target[target == v] = current_max + i
                    target[target > 0] -= current_max
                    target[target < 0] = 0

                target = (
                    F.one_hot(
                        target.to(torch.int64).squeeze(),
                        num_classes=self.hparams.num_multimask_outputs,
                    )
                    .permute(3, 0, 1, 2)
                    .float()
                )
                #  return to B C Z Y X
                target = target.unsqueeze(0)

                with torch.no_grad():
                    i, j = self.matcher(preds, target)
                    # TODO: if the number of masks between preds and targets doesn't match, the discarded masks
                    # could be considered as over/under segmentation regions
                    new_dense_prompts, new_point_prompt = self.compute_prompts(
                        preds[:, i].squeeze(0), target[:, j].squeeze(0)
                    )
                    new_point_prompt, point_type = new_point_prompt
                    # only include new point prompts if prediction is not perfect
                    if new_point_prompt is not None:
                        dense_prompts[ix] = new_dense_prompts
                        if point_prompts[ix] is not None:
                            point_prompts[ix] = torch.cat(
                                (point_prompts[ix], new_point_prompt), dim=1
                            )
                            point_labels[ix] = torch.cat((point_labels[ix], point_type), dim=1)
                        else:
                            point_prompts[ix] = new_point_prompt
                            point_labels[ix] = point_type

                seg_loss += self.loss(preds[:, i], target[:, j])
                with torch.no_grad():
                    # per-mask iou
                    intersection = (preds[:, i].gt(0) & target[:, j].gt(0)).sum(axis=(2, 3, 4))
                    union = (preds[:, i].gt(0) | target[:, j].gt(0)).sum(axis=(2, 3, 4))
                    iou = (intersection + 1e-9) / (union + 1e-9)
                    # print iou by round
                    # print('Epoch:', self.current_epoch, 'Round', round_num, 'iou: ',iou.mean() )
                qc_loss += F.mse_loss(qc, iou)
            seg_loss = seg_loss / images.shape[0]
            qc_loss = qc_loss / images.shape[0]

            loss = seg_loss + qc_loss
            if stage == "train":
                opt.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(opt, gradient_clip_val=0.8, gradient_clip_algorithm="norm")
                if batch_idx % 6 == 0:
                    opt.step()

            if round_num == 0:
                # only update image embeddings based off first round, then only optimize decoder
                image_embeddings = image_embeddings.detach()

                sched = self.lr_schedulers()
                sched.step()

        if batch_idx == 0:
            self.save_images(
                preds[0, i].detach().cpu().numpy(), target[0, j].detach().cpu().numpy()
            )

        return {"loss": loss, "seg_loss": seg_loss, "qc_loss": qc_loss}, None, None

    def predict_step(self, batch, batch_idx):
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
