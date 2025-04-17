import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback

warnings.simplefilter("once", UserWarning)


class SaveTabularData(Callback):
    """Callback to save tabular data to disk as a .csv or .parquet after prediction."""

    def __init__(
        self,
        save_dir,
        meta_keys=[],
        as_parquet: bool = True,
        save_suffix: str = None,
        col_prefix: str = "feat",
    ):
        """
        Parameters
        ----------
        save_dir: str
            directory to save the tabular data
        meta_keys: list
            list of keys in the metadata to include as columns in the saved data
        as_parquet: bool
            Saves data as parquet if True, otherwise saves as csv
        save_suffix: str
            suffix to add to the saved file name
        col_prefix: str
            prefix to add to the column names of the saved data
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.meta_keys = meta_keys
        self.as_parquet = as_parquet
        self.save_suffix = save_suffix
        self.col_prefix = col_prefix

    def pred_to_df(self, pred):
        return pd.DataFrame(pred, columns=[f"{self.col_prefix}_{i}" for i in range(pred.shape[1])])

    def _parse_meta(self, meta):
        """Turn tensors in metadata into numpy arrays and single-element tensors/arrays/lists into
        numbers."""
        for k, v in meta.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    v = v.numpy()
            elif isinstance(v, (list, np.ndarray)) and len(v) == 1:
                v = v[0]
            meta[k] = v
        return meta

    def _save(self, feats, stage):
        save_name = (
            self.save_dir / str(stage)
            if self.save_suffix is None
            else self.save_dir / f"{stage}_{self.save_suffix}"
        )
        if self.as_parquet:
            feats = pd.concat(feats)
            for col in feats.select_dtypes(include=[np.float16]).columns:
                feats[col] = feats[col].astype(np.float32)
            feats.columns = feats.columns.astype(str)
            feats.to_parquet(str(save_name) + ".parquet")
        else:
            pd.concat(feats).to_csv(str(save_name) + ".csv", index=False)

    def save_feats(self, predictions, stage):
        feats = []
        for pred, meta in predictions:
            meta = self._parse_meta(meta)
            batch_feats = self.pred_to_df(pred)
            for k in self.meta_keys:
                if k in meta:
                    batch_feats[k] = meta[k]
                else:
                    warnings.warn(
                        f"Metadata key {k} not found in metadata. Available keys are {meta.keys()}"
                    )
            feats.append(batch_feats)
        self._save(feats, stage)

    def on_predict_epoch_end(self, trainer, pl_module):
        # Access the list of predictions from all predict_steps
        predictions = trainer.predict_loop.predictions
        self.save_feats(predictions, "predict")
