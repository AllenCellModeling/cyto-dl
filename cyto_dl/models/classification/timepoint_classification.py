from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange

from cyto_dl.models.classification import Classification
from cyto_dl.models.utils import find_indices


class TimepointClassification(Classification):
    def __init__(
        self,
        *,
        model: nn.Module,
        x_key: str,
        num_classes: int,
        y_key: str = "label",
        save_dir="./",
        save_movie: bool = True,
        save_images_every_n_epochs=10,
        compile=False,
        write_batch_predictions=False,
        **base_kwargs,
    ):
        super().__init__(
            model=model,
            x_key=x_key,
            num_classes=num_classes,
            y_key=y_key,
            save_dir=save_dir,
            save_images_every_n_epochs=save_images_every_n_epochs,
            compile=compile,
            write_batch_predictions=write_batch_predictions,
            **base_kwargs,
        )

    def predict_step(self, batch, batch_idx):
        x = rearrange(batch[self.hparams.x_key], "b c h w -> c b h w")
        logits = self(x).squeeze(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        if self.hparams.write_batch_predictions:
            pd.DataFrame([preds]).to_csv(
                Path(self.hparams.save_dir) / f"predictions_batch={batch_idx}.csv", index=False
            )
        if self.hparams.save_movie:
            self.save_images(
                batch,
                "predict",
                logits,
                name=str(batch["track_id"].cpu().item()),
            )

        timepoints = np.array(batch["timepoints"][0][1:-1].split(",")).astype(int)
        track_midpoint = (timepoints[0] + timepoints[-1]) // 2

        # breakdowns are transitions from interphase (0) to mitotic (1)
        breakdowns = find_indices(preds, [0, 1])
        # formations are transitions from mitotic (1) to interphase (0)
        # add 1 because the formation index is after index of transition
        formations = find_indices(preds, [1, 0]) + 1

        # -1 -> no formation/breakdown
        if formations.size == 0:
            formation = -1
        else:
            # when multiple formations present, take first, indexing into timepoints
            formation = timepoints[np.min(formations)]
            # formation should occur in the first half of the track
            formation = formation if formation < track_midpoint else -1

        if breakdowns.size == 0:
            breakdown = -1
        else:
            # when multiple breakdowns present, take last, indexing into timepoints
            breakdown = timepoints[np.max(breakdowns)]
            # breakdown should occur in the second half of the track
            breakdown = breakdown if breakdown > track_midpoint else -1

        predictions = {
            "track_id": batch["track_id"].cpu().item(),
            "formation": formation,
            "breakdown": breakdown,
            "timepoints": timepoints,
        }

        return predictions
