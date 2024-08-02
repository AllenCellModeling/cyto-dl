import torch
import torch.nn as nn
from einops import rearrange
from cyto_dl.models.base_model import BaseModel
import pandas as pd
from pathlib import Path
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
        super().__init__(encoder=encoder, predictor=predictor, x_key=x_key, save_dir=save_dir, momentum=momentum, max_epochs=max_epochs, **base_kwargs)

    def model_step(self, stage, batch, batch_idx):
        self.update_teacher()
        source = batch[f'{self.hparams.x_key}_brightfield']
        target = batch[f'{self.hparams.x_key}_struct']

        target_masks = self.get_mask(batch, 'target_mask')
        context_masks = self.get_mask(batch, 'context_mask')
        target_embeddings = self.get_target_embeddings(target, target_masks)
        context_embeddings = self.get_context_embeddings(source, context_masks)
        predictions= self.predictor(context_embeddings, target_masks, batch['structure_name'])

        loss = self.loss(predictions, target_embeddings)
        return loss, None, None

    def get_predict_masks(self, batch_size, num_patches=[4, 16, 16]):
        mask = torch.ones(num_patches, dtype=bool)
        mask = rearrange(mask, 'z y x -> (z y x)')
        mask = torch.argwhere(mask).squeeze()

        return repeat(mask, 't -> t b', b=batch_size)
    
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

