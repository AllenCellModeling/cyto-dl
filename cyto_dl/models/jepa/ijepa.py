import torch
import torch.nn as nn
from torchmetrics import MeanMetric
from einops import rearrange
from cyto_dl.models.base_model import BaseModel
import copy
from cyto_dl.nn.vits.utils import take_indexes
import pandas as pd
from pathlib import Path
import numpy as np
from einops import repeat

class IJEPA(BaseModel):
    def __init__(
            self,
            *,
            encoder: nn.Module,
            predictor: nn.Module,
            x_key: str,
            save_dir: str= './',
            momentum: float=0.998,
            max_epochs: int=100,
            **base_kwargs,
        ):
            """
            Initialize the IJEPA model.

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

    def forward(self, x):
        return self.encoder(x)
    
    def _get_momentum(self):
        # linearly increase the momentum from self.momentum to 1 over course of self.hparam.max_epochs
        return self.hparams.momentum + (1 - self.hparams.momentum) * self.current_epoch / self.hparams.max_epochs
    
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
            target_embeddings= self.teacher(x)
        target_embeddings = rearrange(target_embeddings, "b t c -> t b c")
        target_embeddings = take_indexes(target_embeddings, mask)
        target_embeddings = rearrange(target_embeddings, "t b c -> b t c")
        return target_embeddings

    def get_context_embeddings(self, x, mask):
        # mask context pre-embedding to prevent leakage of target information
        context_patches, _, _, _ = self.encoder.patchify(x, 0)
        context_patches = take_indexes(context_patches, mask)
        context_patches= self.encoder.transformer_forward(context_patches)
        return context_patches
    
    def get_mask(self, batch, key):
        return rearrange(batch[key], "b t -> t b")

    #IWM
    # def model_step(self, stage, batch, batch_idx):
    #     self.update_teacher()
    #     source = batch[f'{self.hparams.x_key}_brightfield']
    #     target = batch[f'{self.hparams.x_key}_struct']

    #     target_masks = self.get_mask(batch, 'target_mask')
    #     context_masks = self.get_mask(batch, 'context_mask')
    #     target_embeddings = self.get_target_embeddings(target, target_masks)
    #     context_embeddings = self.get_context_embeddings(source, context_masks)
    #     predictions= self.predictor(context_embeddings, target_masks, batch['structure_name'])

    #     loss = self.loss(predictions, target_embeddings)
    #     return loss, None, None

    # ijepa
    def model_step(self, stage, batch, batch_idx):
        self.update_teacher()
        input=batch[self.hparams.x_key]

        target_masks = self.get_mask(batch, 'target_mask')
        context_masks = self.get_mask(batch, 'context_mask')

        target_embeddings = self.get_target_embeddings(input, target_masks)
        context_embeddings = self.get_context_embeddings(input, context_masks)
        predictions= self.predictor(context_embeddings, target_masks)

        loss = self.loss(predictions, target_embeddings)
        return loss, None, None

    def get_predict_masks(self, batch_size, num_patches=[4, 16, 16]):
        mask = torch.ones(num_patches, dtype=bool)
        mask = rearrange(mask, 'z y x -> (z y x)')
        mask = torch.argwhere(mask).squeeze()

        return repeat(mask, 't -> t b', b=batch_size)
    
    # IWM
    def predict_step(self, batch, batch_idx):
        source = batch[f'{self.hparams.x_key}_brightfield'].squeeze(0)
        target = batch[f'{self.hparams.x_key}_struct'].squeeze(0)

        # use predictor to predict each patch
        target_masks = self.get_predict_masks(source.shape[0])
        # mean across patches
        bf_embeddings = self.encoder(source)
        pred_target_embeddings = self.predictor(bf_embeddings, target_masks, batch['structure_name']).mean(axis=1)
        pred_feats = pd.DataFrame(pred_target_embeddings.detach().cpu().numpy(), columns=[f'{i}_pred' for i in range(pred_target_embeddings.shape[1])])

        # get target embeddings
        target_embeddings = self.encoder(target).mean(axis=1)
        ctxt_feats = pd.DataFrame(target_embeddings.detach().cpu().numpy(), columns=[f'{i}_ctxt' for i in range(target_embeddings.shape[1])])

        all_feats = pd.concat([ctxt_feats, pred_feats], axis=1)

        all_feats.to_csv(Path(self.hparams.save_dir) / f"{batch_idx}_predictions.csv")
        return None, None, None

    # def predict_step(self, batch, batch_idx):
    #     x=batch[self.hparams.x_key]
    #     embeddings = self(x).mean(axis=1)
    #     preds = pd.DataFrame(embeddings.detach().cpu().numpy(), columns=[str(i) for i in range(embeddings.shape[1])])
    #     preds['CellId'] =batch['CellId']
    #     preds.to_csv(Path(self.hparams.save_dir) / f"{batch_idx}_predictions.csv")
    #     return None, None, None
