from pathlib import Path

import pandas as pd
from lightning.pytorch.callbacks import Callback


class CSVSaver(Callback):
    def __init__(self, save_dir, meta_keys=[]):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.meta_keys = meta_keys

    def pred_to_csv(self, pred):
        return pd.DataFrame(pred)

    def save_feats(self, predictions, stage):
        feats = []
        for pred, meta in predictions:
            batch_feats = self.pred_to_csv(pred)
            for k in self.meta_keys:
                batch_feats[k] = meta[k]
            feats.append(batch_feats)
        pd.concat(feats).to_csv(self.save_dir / f"{stage}.csv", index=False)

    def on_predict_epoch_end(self, trainer, pl_module):
        # Access the list of predictions from all predict_steps
        predictions = trainer.predict_loop.predictions
        self.save_feats(predictions, "predict")


class JEPASaver(CSVSaver):
    def pred_to_csv(self, pred):
        source_embed, target_embed, pred_target_embed = pred

        source_feats = pd.DataFrame(source_embed)
        source_feats["feat_type"] = "source"

        target_feats = pd.DataFrame(target_embed)
        target_feats["feat_type"] = "target"

        pred_feats = pd.DataFrame(pred_target_embed)
        pred_feats["feat_type"] = "pred"

        all_feats = pd.concat([source_feats, target_feats, pred_feats])
        return all_feats
