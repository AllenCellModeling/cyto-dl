from typing import Optional
from warnings import warn

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cyto_dl.models.im2im.utils.postprocessing import detach


class DiffAELatentWalk(Callback):
    def __init__(
        self,
        num_pcs: int = 8,
        n_steps: int = 10,
        range: Optional[int] = None,
        every_n_epoch: int = 1,
    ):
        """
        Parameters
        ----------
        num_pcs: int=8
            Number of principal components to use for latent walk
        n_steps: int=10
            Number of steps to traverse each PC in the latent walk
        range: Optional[int]=None
            Range to traverse each PC in the latent walk. If None, the min and max of the PC are used.
        every_n_epoch:int=1
            Frequency to perform latent walk
        """
        self.num_pcs = num_pcs
        self.n_steps = n_steps
        self.range = range
        self.every_n_epoch = every_n_epoch

        self.pca = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=num_pcs))])

        self.val_feats = []

    def _latent_walk(self, feats, model, stage):
        # catch if only one batch for validation
        if len(feats.shape) == 1 or feats.shape[0] < self.num_pcs:
            warn(f"Insufficient data for latent walk with {self.num_pcs} PCs. Skipping...")
            return
        pca_data = self.pca.fit_transform(feats)
        print(f"Explained variance ratio: {self.pca['pca'].explained_variance_ratio_}")
        walk = []
        for pc in np.arange(self.num_pcs):
            std = pca_data[:, pc].std()
            if self.range is None:
                min = pca_data[:, pc].min() / std
                max = pca_data[:, pc].max() / std
                range_ = np.linspace(min, max, self.n_steps)
            else:
                range_ = np.linspace(-self.range, self.range, self.n_steps)
            for i in range_:
                array = np.zeros(self.num_pcs)
                array[pc] = i * std
                walk.append(array)
        walk = np.stack(walk).squeeze()
        walk = self.pca.inverse_transform(walk)
        walk = torch.from_numpy(walk).float().to(model.device)
        model.latent_walk(walk, stage)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if (trainer.current_epoch + 1) % self.every_n_epoch == 0:
            self.val_feats.append(detach(outputs[1].squeeze()))

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epoch == 0:
            feats = np.concatenate(self.val_feats)
            self._latent_walk(feats, trainer.model, f"{trainer.current_epoch}_val_latent_walk")

    def on_predict_epoch_end(self, trainer, pl_module):
        feats = np.concatenate([x[0] for x in trainer.predict_loop.predictions])
        self._latent_walk(feats, trainer.model, "latent_walk")
