import copy

import torch
import torch.nn as nn
from einops import rearrange
from torchmetrics import MeanMetric

from cyto_dl.models.base_model import BaseModel
from cyto_dl.nn.vits.utils import take_indexes


class JEPABase(BaseModel):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        predictor: nn.Module,
        x_key: str,
        save_dir: str = "./",
        momentum: float = 0.998,
        max_epochs: int = 100,
        **base_kwargs,
    ):
        """Base for IJEPA and IWM models.

        Parameters
        ----------
        encoder : nn.Module
            The encoder module used for feature extraction.
        predictor : nn.Module
            The predictor module used for generating predictions.
        x_key : str
            The key used to access the input data.
        momentum : float, optional
            The momentum value for the exponential moving average of the model weights (default is 0.998).
        max_epochs : int, optional
            The maximum number of training epochs (default is 100).
        **base_kwargs : dict
            Additional arguments passed to the BaseModel.
        """
        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
        }
        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.encoder = encoder
        self.predictor = predictor

        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.loss = torch.nn.L1Loss()

    def configure_optimizers(self):
        optimizer = self.optimizer(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
        )
        scheduler = self.lr_scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1,
            },
        }

    def remove_first_dim(self, tensor):
        # account for grid patching transform
        return tensor.squeeze(0) if len(tensor.shape) == 6 else tensor

    def forward(self, x):
        return self.encoder(x)

    def _get_momentum(self):
        # linearly increase the momentum from self.momentum to 1 over course of self.hparam.max_epochs
        return (
            self.hparams.momentum
            + (1 - self.hparams.momentum) * self.current_epoch / self.hparams.max_epochs
        )

    # modified from https://github.com/facebookresearch/jepa/blob/main/app/vjepa/train.py
    def update_teacher(self):
        # ema of teacher
        momentum = self._get_momentum()
        # momentum update of the parameters of the teacher network
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.teacher.parameters()):
                param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def get_target_embeddings(self, x, mask):
        # embed the target with full context for maximally informative embeddings
        with torch.no_grad():
            target_embeddings = self.teacher(x)
        target_embeddings = rearrange(target_embeddings, "b t c -> t b c")
        target_embeddings = take_indexes(target_embeddings, mask)
        target_embeddings = rearrange(target_embeddings, "t b c -> b t c")
        return target_embeddings

    def get_context_embeddings(self, x, mask):
        # mask context pre-embedding to prevent leakage of target information
        context_patches, _, _, _ = self.encoder.patchify(x, 0)
        context_patches = take_indexes(context_patches, mask)
        context_patches = self.encoder(context_patches, patchify=False)
        return context_patches

    def get_mask(self, batch, key):
        return rearrange(batch[key], "b t -> t b")

    def model_step(self, stage, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx):
        x = batch[self.hparams.x_key]
        x = self.remove_first_dim(x)

        embeddings = self(x)

        return embeddings.detach().cpu().numpy(), x.meta
