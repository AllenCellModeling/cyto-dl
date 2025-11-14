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
        for _, batch in predictions:
            meta, pred = batch
            batch_feats = {key: [val.replace('\0','') if isinstance(val,str) else val] for key,val in meta.items() if key not in ['roi', 'Link Path']}
            # batch_feats['features'] = [pred.tolist()]
            for patch, roi, link in zip(pred.tolist(), meta['roi'], meta['Link Path']):
                patch_feats = batch_feats.copy()
                patch_feats['features'] = [patch]
                patch_feats['roi'] = [roi]
                patch_feats['Link Path'] = [link]
                patch_feats = pd.DataFrame(patch_feats)
                feats.append(patch_feats)
        self._save(feats, stage)

    def on_predict_epoch_end(self, trainer, pl_module):
        # Access the list of predictions from all predict_steps
        predictions = trainer.predict_loop.predictions
        # import pdb; pdb.set_trace()
        self.save_feats(predictions, "predict")
