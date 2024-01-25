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
            # "train/loss/track": MeanMetric(),
            # "val/loss/track": MeanMetric(),
            # "train/loss/tp": MeanMetric(),
            # "val/loss/tp": MeanMetric(),
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

        self.regression_loss = nn.MSELoss()

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
            label = batch["label"].squeeze(0) #.argmax(dim=1) # argmax only needed with soft labels
        # save labeled movie of prediction
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.load_default()
        # Define text positions
        movie = []
        for p, l, frame in zip(pred, label, batch[self.hparams.x_key].cpu().numpy().squeeze()):
            # add il and wl text to frame
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
        return loss, logits.argmax(dim=1), labels #.argmax(dim=1)


    # def model_step(self, stage, batch, batch_idx):
    #     logits, tp_pred = self(batch[self.hparams.x_key])
    #     logits = logits.squeeze(0)
    #     labels = batch["tp_label"].squeeze(0)
    #     if self.should_save_image(batch_idx, stage):
    #         print(tp_pred, batch['window_labels'].float())
    #         self.save_images(batch, stage, logits)
    #     per_tp_loss = self.loss_fn(logits, labels.long())

    #     regression_loss = self.regression_loss(tp_pred, batch['window_labels'].float()) 
    #     current_step = min(self.current_epoch, 300)
    #     value = (1 - math.cos(math.pi * current_step / 300)) / 2.0
    #     regression_loss = value * regression_loss
    #     # if batch['window_labels'][0, 0]>=0:
    #         # regression_loss[0, 0] *= 40
    #     # regression_loss = regression_loss.mean()

    #     loss = {
    #         'tp': per_tp_loss,
    #         'regression': regression_loss,
    #         'loss': per_tp_loss + regression_loss
    #     }

    #     return loss, logits.argmax(dim=1), labels #.argmax(dim=1)

    #  classification across track
    # def model_step(self, stage, batch, batch_idx):
    #     logits, pred_form, pred_break = self(batch[self.hparams.x_key])
    #     logits = logits.squeeze(0)
    #     labels = batch['tp_label'].squeeze(0).long()
    #     if self.should_save_image(batch_idx, stage):
    #         with torch.no_grad():
    #             name = f"{self.current_epoch}_pred_form={torch.argmax(pred_form, dim=1).cpu().item()}_pred_break={torch.argmax(pred_break, dim=1).cpu().item()}"
    #         self.save_images(batch, stage, logits, name=name)
    #     loss_class = self.loss_fn(logits, labels)

    #     loss_track = self.track_classification_loss(pred_form.squeeze(0), batch['formation_label'].squeeze(0).float()) + self.track_classification_loss(pred_break.squeeze(0), batch['breakdown_label'].squeeze(0).float())
    #     loss  = {
    #         'tp': loss_class,
    #         'track': loss_track,
    #         'loss': loss_class + loss_track
    #     }
    #     return loss, logits, labels

    def plot_labels(self, labels, name):
        fig, ax = plt.subplots()
        ax.plot(labels[:, 0], label="interphase")
        ax.plot(labels[:, 1], label="formation")
        ax.plot(labels[:, 2], label="breakdown")
        ax.plot(labels[:, 3], label="mitotic")
        ax.legend()
        fig.savefig(Path(self.hparams.save_dir) / "plots" / f"{name}.png")
        plt.close(fig)

    # def plot(self, batch, probs):
    #     start = batch['track_start'][0].cpu()
    #     track_id = batch['track_id'][0].cpu()
    #     formation = batch['formation'][0].cpu()
    #     breakdown = batch['breakdown'][0].cpu()

    #     fig, ax = plt.subplots()
    #     ax.plot(probs[:, 1],label ='formation probs')
    #     ax.plot(probs[:, 2],label ='breakdown probs')
    #     if formation >=0:
    #         ax.vlines(formation - start, 0, 1, color='r', label='formation')
    #     if breakdown >=0:
    #         ax.vlines(breakdown - start, 0, 1, color='g', label='breakdown')
    #     ax.legend()
    #     fig.savefig(Path(self.hparams.save_dir)/ 'plots' / f'{track_id}.png')
    #     plt.close(fig)

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

        breakdowns = self.find_indices(preds, [0,1])
        formations = self.find_indices(preds, [1, 0])
        print(formations, breakdowns)
        if formations.size==0:
            formation = -1
        else:
            formation = np.min(formations)

        if breakdowns.size==0:
            breakdown = -1
        else:
            breakdown = np.max(breakdowns)

        self.predictions.append({
            'movie': batch['movie'][0],
            'track_id': track_id,
            'formation': formation + track_start if formation >=0 else -1,
            'breakdown': breakdown + track_start if breakdown >=0 else -1,
        })


    # def predict_step(self, batch, batch_idx):
    #     logits = self(batch[self.hparams.x_key]).squeeze(0)

    #     if self.hparams.save_movie:
    #         self.save_images(
    #             batch,
    #             "predict",
    #             logits,
    #             name=f"{batch['movie'][0]}_{batch['track_id'].cpu().item()}",
    #         )

    #     preds = torch.argmax(logits, dim=1).cpu().numpy()

    #     # find uncaught breakdowns where there is a transition from 0 to 3
    #     additional_breakdowns = self.find_indices(preds, [0,3])
    #     preds[additional_breakdowns] = 2

    #     # find uncaught formations where there is a transition from 3 to 0
    #     additional_formations = self.find_indices(preds, [3,0])
    #     preds[additional_formations + 1] = 1

    #     track_start = batch["track_start"].cpu().item()
    #     track_id = batch["track_id"].cpu().item()

    #     formation_pred = np.where(preds == 1)[0]
    #     if len(formation_pred) == 0:
    #         formation_pred = -1
    #     else:
    #         formation_pred += track_start

    #     breakdown_pred = np.where(preds == 2)[0]
    #     if len(breakdown_pred) == 0:
    #         breakdown_pred = -1
    #     else:
    #         breakdown_pred += track_start

    #     self.predictions = pd.concat(
    #         [
    #             self.predictions,
    #             pd.DataFrame(
    #                 {
    #                     "movie": [batch["movie"][0]],
    #                     "track_id": [track_id],
    #                     "preds": [str(preds)],
    #                     "predicted_formation": [formation_pred],
    #                     "predicted_breakdown": [breakdown_pred],
    #                     "track_length": [batch[self.hparams.x_key].shape[1]],
    #                 }
    #             ),
    #         ]
    #     )
    #     with open(Path(self.hparams.save_dir) / "predictions.csv", 'a') as f:
    #         self.predictions.to_csv(f, header=f.tell()==0, index=False)
    #     return logits

    def test_step(self, batch, batch_idx):
        logits = self(batch[self.hparams.x_key]).squeeze(0)
        labels = batch["label"].squeeze(0)

        self.plot_labels(
            labels.cpu().numpy(), f"{batch['movie'][0]}_{batch['track_id'].cpu().item()}"
        )
        if self.hparams.save_movie:
            self.save_images(
                batch,
                "predict",
                logits,
                name=f"{batch['movie'][0]}_{batch['track_id'].cpu().item()}",
            )

        loss = self.loss_fn(logits, labels)

        track_start = batch["track_start"].cpu().item()
        track_id = batch["track_id"].cpu().item()
        formation = batch["formation"].cpu().item()
        breakdown = batch["breakdown"].cpu().item()

        # predicted_timepoints = torch.argmax(logits, dim = 0).detach().cpu().numpy()
        # predicted_formation_idx, predicted_breakdown_idx = predicted_timepoints[1], predicted_timepoints[2]
        # per_tp_probs= torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        # formation_prob = per_tp_probs[predicted_formation_idx, 1]
        # breakdown_prob = per_tp_probs[predicted_breakdown_idx, 2]

        # formation_pred = predicted_formation_idx + track_start if formation_prob > 0.5 else -1
        # breakdown_pred = predicted_breakdown_idx + track_start if breakdown_prob > 0.5 else -1

        preds = torch.argmax(logits, dim=1).cpu().numpy()

        # find uncaught breakdowns where there is a transition from 0 to 3
        additional_breakdowns = self.find_indices(preds, 0, 3)
        preds[additional_breakdowns] = 2

        # find uncaught formations where there is a transition from 3 to 0
        additional_formations = self.find_indices(preds, 3, 0)
        preds[additional_formations] = 1

        formation_pred = np.where(preds == 1)[0]
        if len(formation_pred) == 0:
            formation_pred = -1
        else:
            formation_pred += track_start

        breakdown_pred = np.where(preds == 2)[0]
        if len(breakdown_pred) == 0:
            breakdown_pred = -1
        else:
            breakdown_pred += track_start

        self.predictions = pd.concat(
            [
                self.predictions,
                pd.DataFrame(
                    {
                        "movie": [batch["movie"][0]],
                        "track_id": [track_id],
                        "loss": [loss.cpu()],
                        "preds": [str(preds)],
                        "labels": [str(labels.argmax(dim=1).cpu().numpy())],
                        "predicted_formation": [formation_pred],
                        "predicted_breakdown": [breakdown_pred],
                        "formation": [formation],
                        "breakdown": [breakdown],
                        "track_length": [batch["label"].shape[1]],
                    }
                ),
            ]
        )
        with open(Path(self.hparams.save_dir) / "predictions.csv", 'a') as f:
            self.predictions.to_csv(f, header=f.tell()==0, index=False)
        return logits
