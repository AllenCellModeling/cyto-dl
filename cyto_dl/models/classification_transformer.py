import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import math
import torch.nn as nn
from aicsimageio.writers import OmeTiffWriter
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score

from cyto_dl.models.base_model import BaseModel
from PIL import Image, ImageDraw, ImageFont


class Classifier(BaseModel):
    def __init__(
        self,
        *,
        model: nn.Module,
        x_key: str,
        num_classes: int,
        save_dir="./",
        save_movie: bool = True,
        save_images_every_n_epochs=10,
        compile=False,
        **base_kwargs,
    ):
        """
        Parameters
        ----------
        model: nn.Module
            model network, parameters are shared between task heads
        x_key: str
            key of input image in batch
        save_dir="./"
            directory to save images during training and validation
        save_images_every_n_epochs=1
            Frequency to save out images during training
        compile: False
            Whether to compile the model using torch.compile
        **base_kwargs:
            Additional arguments passed to BaseModel
        """
        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
            "train/f1": MulticlassF1Score(num_classes=num_classes),
            "val/f1": MulticlassF1Score(num_classes=num_classes),
        }
        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.automatic_optimization = True
        for stage in ("train", "val", "test", "predict"):
            (Path(save_dir) / f"{stage}_images").mkdir(exist_ok=True, parents=True)
        (Path(save_dir) / "plots").mkdir(exist_ok=True, parents=True)
        if compile is True and not sys.platform.startswith("win"):
            self.model = torch.compile(model)
        else:
            self.model = model

        self.loss_fn = nn.CrossEntropyLoss()

        self.predictions = []


    def forward(self, x):
        return self.model(x)

    def should_save_image(self, batch_idx, stage):
        return stage in ("test", "predict") or (self.current_epoch + 1) % self.hparams.save_images_every_n_epochs == batch_idx ==0
        

    def save_images(self, batch, stage, logits, name=None):
        pred = torch.argmax(logits, dim=1)

        if stage == "predict":
            label = torch.zeros_like(pred)
        else:
            label = batch["label"].squeeze(0)
        # save labeled movie of prediction

        font = ImageFont.load_default()
        # Define text positions
        movie = []
        for p, l, frame in zip(pred, label, batch[self.hparams.x_key].cpu().numpy().squeeze()):
            # Convert the NumPy array to a PIL Image
            image = Image.fromarray(frame.astype(float))

            # Create a drawing context
            draw = ImageDraw.Draw(image)
            # Draw text on the image
            draw.text((0, 0), f"P:{p}", fill=2 * frame.max(), font=font)
            if stage != "predict":
                draw.text((0, 20), f"L:{int(l)}", fill=2 * frame.max(), font=font)

            # Convert the image back to a NumPy array
            movie.append(np.array(image))

        if name is None:
            name = self.current_epoch

        movie = rescale_intensity(np.stack(movie), out_range=(0, 255)).astype(np.uint8)

        OmeTiffWriter.save(
            uri=Path(self.hparams.save_dir) / f"{stage}_images" / f"{name}.ome.tiff",
            data=movie,
        )

    def model_step(self, stage, batch, batch_idx):
        logits = self(batch[self.hparams.x_key]).squeeze(0)
        labels = batch["label"].squeeze(0)
        if self.should_save_image(batch_idx, stage):
            self.save_images(batch, stage, logits)
        loss = self.loss_fn(logits, labels.long())
        return loss, logits.argmax(dim=1), labels

    def find_indices(self, lst, vals):
        arr = np.array(lst)

        sets = []
        for i, val in enumerate(vals):
            indices = set(np.where(arr == val)[0] - i)
            sets.append(indices)
        return np.asarray(list(set.intersection(*sets)), dtype=int)

    def on_predict_epoch_end(self):
        pd.DataFrame(self.predictions).to_csv(Path(self.hparams.save_dir) / "predictions.csv", index=False)

    def predict_step(self, batch, batch_idx):
        logits = self(batch[self.hparams.x_key]).squeeze(0)
        if self.hparams.save_movie:
            self.save_images(
                batch,
                "predict",
                logits,
                name=f"{batch['movie'][0]}_{batch['track_id'].cpu().item()}",
            )
    
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        track_start = batch["track_start"].cpu().item()
        track_id = batch["track_id"].cpu().item()

        # breakdowns are transitions from interphase (0) to mitotic (1)
        breakdowns = self.find_indices(preds, [0,1])
        # formations are transitions from mitotic (1) to interphase (0)
        formations = self.find_indices(preds, [1, 0])

        # -1 -> no formation/breakdown
        if formations.size==0:
            formation = -1
        else:
            # when multiple formations present, take first
            formation = np.min(formations)

        if breakdowns.size==0:
            breakdown = -1
        else:
            # when multiple breakdowns present, take last
            breakdown = np.max(breakdowns)

        self.predictions.append({
            'movie': batch['movie'][0],
            'track_id': track_id,
            'formation': formation + track_start if formation >=0 else -1,
            'breakdown': breakdown + track_start if breakdown >=0 else -1,
        })