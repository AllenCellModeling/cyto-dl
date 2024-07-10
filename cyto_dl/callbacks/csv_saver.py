from pathlib import Path

import pandas as pd
from lightning.pytorch.callbacks import Callback


class CSVSaver(Callback):
    def __init__(self, save_dir, meta_keys=[]):
        self.save_dir = Path(save_dir)
        self.meta_keys = meta_keys

    def on_predict_epoch_end(self, trainer, pl_module):
        # Access the list of predictions from all predict_steps
        predictions = trainer.predict_loop.predictions
        feats = []
        for pred, meta in predictions:
            batch_feats = pd.DataFrame(pred)
            batch_feats["filename"] = meta["filename_or_obj"]
            feats.append(batch_feats)
        pd.concat(feats).to_csv(self.save_dir / "predictions.csv", index=False)
