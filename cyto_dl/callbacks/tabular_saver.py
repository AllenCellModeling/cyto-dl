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
        batch_size=1,
        meta_keys=None,
        as_parquet: bool = True,
        save_suffix: str = None,
        col_prefix: str = "feat",
    ):
        """
        Parameters
        ----------
        save_dir: str
            directory to save the tabular data
        batch_size: int
            batch size of the dataloader
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
        self.meta_keys = meta_keys or []
        self.batch_size = batch_size
        self.as_parquet = as_parquet
        self.save_suffix = save_suffix
        self.col_prefix = col_prefix

    def pred_to_df(self, pred):
        return pd.DataFrame(pred, columns=[f"{self.col_prefix}_{i}" for i in range(pred.shape[1])])

    def _parse_meta(self, raw_meta: dict, num_patches, total_rows) -> dict:
        """
        For each meta_key, repeat each entry num_patches times
        """
        meta = {}

        for k, v in raw_meta.items():
            # basic conversion to python/numpy
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
                if v.numel() == 1:
                    v = v.item()
                else:
                    v = v.numpy()
            elif isinstance(v, (list, np.ndarray)) and len(v) == 1:
                v = v[0]

            if k in self.meta_keys:
                arr = np.array(v)
                if arr.ndim == 0:
                    # scalar: just broadcast
                    exp = np.full(shape=(total_rows,), fill_value=arr.item())
                elif arr.ndim == 1: # and arr.shape[0] == self.batch_size:
                    # repeat each element num_patches times
                    exp = np.repeat(arr, repeats=num_patches)
                
                meta[k] = exp
            else:
                # we don't need to expand keys we won't write out
                continue

        return meta

    def _save(self, feats, stage: str):
        save_name = (
            self.save_dir / str(stage)
            if self.save_suffix is None
            else self.save_dir / f"{stage}_{self.save_suffix}"
        )
        df = pd.concat(feats, ignore_index=True)
        if self.as_parquet:
            # parquet doesn't support float16 well
            for c in df.select_dtypes(include=[np.float16]).columns:
                df[c] = df[c].astype(np.float32)
            df.columns = df.columns.astype(str)
            df.to_parquet(f"{save_name}.parquet", index=False)
        else:
            df.to_csv(f"{save_name}.csv", index=False)


    def save_feats(self, predictions, stage: str):
        feats = []
        num_patches_set = set()
        for pred, raw_meta in predictions:
            batch_feats = self.pred_to_df(pred)
            num_patches_set.add(len(batch_feats)//self.batch_size)
            num_patches = list(num_patches_set)[0] # Since we always need to return just the first element here!
            total_rows = len(batch_feats)
            meta = self._parse_meta(raw_meta, num_patches, total_rows)
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
        predictions = trainer.predict_loop.predictions
        self.save_feats(predictions, "predict")
