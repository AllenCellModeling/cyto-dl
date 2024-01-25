import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score

from cyto_dl.models.base_model import BaseModel
from aicsimageio.writers import OmeTiffWriter
import numpy as np
from monai.inferers import SlidingWindowSplitter
from monai.networks.blocks import MLPBlock
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity

import math
from einops import rearrange

import pandas as pd
from skimage.measure import label 


class Classifier(BaseModel):
    def __init__(
        self,
        *,
        model: nn.Module,
        x_key: str,
        window_size: int = 3,
        save_dir="./",
        save_movie:bool = True,
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
            "train/loss/instance": MeanMetric(),
            "val/loss/instance": MeanMetric(),
            "train/loss/window": MeanMetric(),
            "val/loss/window": MeanMetric(),
            "test/loss": MeanMetric(),
            "train/f1/instance": MulticlassF1Score(num_classes = 4),
            "train/f1/window": MulticlassF1Score(num_classes = window_size),
            "val/f1/instance": MulticlassF1Score(num_classes = 4),
            "val/f1/window": MulticlassF1Score(num_classes = window_size),
        }
        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.automatic_optimization = True
        for stage in ("train", "val", "test", "predict"):
            (Path(save_dir) / f"{stage}_images").mkdir(exist_ok=True, parents=True)
        (Path(save_dir)/ 'plots').mkdir(exist_ok=True, parents=True)
        if compile is True and not sys.platform.startswith("win"):
            self.model = torch.compile(model)
        else:
            self.model = model

        self.classifier_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(12, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        self.class_loss_fn = nn.CrossEntropyLoss()
        self.window_loss_fn = nn.CrossEntropyLoss()

        self.index_loss_fn = nn.MSELoss()

        self.inference_splitter= SlidingWindowSplitter(
            patch_size=(window_size, 64, 64),
            overlap = window_size-1,
            pad_mode='replicate'
        )

        self.counter = {0:0, 1:0, 2:0}
        self.index_sched_steps = 100000

        self.predictions = pd.DataFrame({
            'track_id': [],
            'predicted_formation': [],
            'predicted_breakdown': [],
            'formation': [],
            'breakdown': []
        })

    def forward(self, x):
        return self.model(x)

    def should_save_image(self, batch_idx, stage):
        return stage in ("test", "predict") or  (self.current_epoch + 1) % self.hparams.save_images_every_n_epochs == 0

    def save_images(self, batch, stage, logits):
        preds = torch.argmax(logits, dim=1)
        for label_val, name in {0: "normal", 1: "formation", 2: "breakdown"}.items():
            img = batch[self.hparams.x_key][preds == label_val]
            if img.shape[0] == 0:
                continue
            img = img.detach().cpu().numpy().astype(np.float32)
            OmeTiffWriter.save(uri = 
                Path(self.hparams.save_dir) / f"{stage}_images" / f"{self.current_epoch}_pred_{name}.ome.tiff",
                data = img
            )

    def cosine_scheduler(self):
        # Ensure current_step is within the range of total_steps
        current_step = min(self.global_step, self.index_sched_steps)
        
        # Calculate the value based on cosine scheduler
        value = (1 - math.cos(math.pi * current_step / self.index_sched_steps)) / 2.0
        
        return value


    # def model_step(self, stage, batch, batch_idx):
    #     #     for lab in batch['label']:
    #     #         self.counter[lab.item()] += 1

    #     logits = self(batch[self.hparams.x_key])
    #     window_classification_logits = logits[:, :3]
    #     index_logits = logits[:, 3:]
    #     if self.should_save_image(batch_idx, stage):
    #         self.save_images(batch, stage, window_classification_logits)
    #     loss_classification = self.loss_fn(window_classification_logits, batch["label"])
    #     loss_index = self.index_loss_fn(index_logits, batch["index"].float()) * self.cosine_scheduler()
    #     loss = {'classification': loss_classification, 'index': loss_index, 'loss': loss_classification + loss_index}
    #     if self.global_step % 1000 == 0:
    #         print('-'*40)
    #         print('COUNTER')
    #         print(self.counter)
    #         print('INDEX')
    #         print(index_logits[:10])
    #         print(batch['index'][:10])
    #         print('-'*40)

    #     return loss, window_classification_logits, batch["label"]

    def model_step(self, stage, batch, batch_idx):
        # for lab in batch['label']:
        #     self.counter[lab.item()] += 1
        instance_logits = self(batch[self.hparams.x_key]) # B x 12 
        window_logits = self.classifier_head(instance_logits) # B x 3

        instance_logits = rearrange(instance_logits,  'b (num_classes window_size ) -> b  num_classes window_size', num_classes = 4, window_size=self.hparams.window_size)

        if batch_idx == 0:
            # print(self.counter)
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.load_default()
            with torch.no_grad():
                movie = []
                instance_preds = torch.argmax(instance_logits, dim=1).clone().detach().cpu().numpy()
                window_preds = torch.argmax(window_logits, dim=1).clone().detach().cpu().numpy()

                for il, wl, frame in zip(instance_preds, window_preds, batch[self.hparams.x_key].cpu().numpy().squeeze()):
                    for i in range(self.hparams.window_size):
                        image = Image.fromarray(frame[i].astype(float))
                        draw = ImageDraw.Draw(image)
                        # Define text positions
                        x1, y1 = 0, 0
                        x2, y2 = 0, 20
                        # Draw text on the image
                        draw.text((x1, y1), f'i:{il[i]}', fill='white', font=font)
                        draw.text((x2, y2), f'w:{wl}', fill='white', font=font)

                        # Convert the image back to a NumPy array
                        movie.append(np.array(image))

                OmeTiffWriter.save(uri = Path(self.hparams.save_dir) / f"{stage}_images" / f"{self.global_step}.ome.tiff", data = np.stack(movie))

        instance_loss = self.class_loss_fn(instance_logits, batch['tp_label'].long())
        window_loss = self.window_loss_fn(window_logits, batch["label"].long())
        loss = {'instance': instance_loss, 'window': window_loss, 'loss': instance_loss + window_loss}
        logits = {'instance': instance_logits, 'window': window_logits}
        label = {'instance': batch['tp_label'].long(), 'window': batch['label'].long()}
        if torch.isnan(loss['loss']):
            breakpoint()
        return loss, logits, label

    def plot(self, batch, probs):
        start = batch['track_start'][0].cpu()
        track_id = batch['track_id'][0].cpu()
        formation = batch['formation'][0].cpu()
        breakdown = batch['breakdown'][0].cpu()

        fig, ax = plt.subplots()
        ax.plot(probs[:, 1],label ='formation probs')
        ax.plot(probs[:, 2],label ='breakdown probs')
        if formation >=0:
            ax.vlines(formation - start, 0, 1, color='r', label='formation')
        if breakdown >=0:
            ax.vlines(breakdown - start, 0, 1, color='g', label='breakdown')
        ax.legend()
        fig.savefig(Path(self.hparams.save_dir)/ 'plots' / f'{track_id}.png')
        plt.close(fig)

    def save_movie(self, batch):
        movie = batch[self.hparams.x_key].detach().cpu().numpy()
        movie = rescale_intensity(movie, out_range=(0, 255)).astype(np.uint8)
        OmeTiffWriter.save(uri = Path(self.hparams.save_dir) / f"movie_{batch['track_id'][0].cpu()}.ome.tiff", data = movie)


    # def predict_step(self, batch, batch_idx):
    #     inputs = batch[self.hparams.x_key]

    #     #duplicate first and last time points
    #     inputs = torch.cat([inputs[:,0:1], inputs, inputs[:,-1:]], dim=1).unsqueeze(0)

    #     logits= torch.stack([self(patch.squeeze(0)) for patch, _ in self.inference_splitter(inputs)]).squeeze(1)
    #     probs= torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
    #     self.plot(batch, probs)
    #     if self.hparams.save_movie:
    #         self.save_movie(batch)

    #     return logits

            #     for lab in batch['label']:
        #         self.counter[lab.item()] += 1



    def instance_consensus(self, instance_acts, track_start):
        from scipy.stats import mode
        pred_1 = instance_acts[:, 0]
        pred_2 = np.pad(instance_acts[:-1, 1], (1,0))
        pred_3 = np.pad(instance_acts[:-2, 2], (2,0))
        comb = np.stack([pred_1, pred_2, pred_3], axis=1)
        out, count = mode(comb, axis = 1)
        out= out.flatten()
        count = count.flatten()

        formation = np.logical_and(out == 1, count >=2)
        breakdown = np.logical_and(out == 2, count >=2)


        if np.sum( formation) == 0:
            formation_pred = [-1]
        else:
            formation_windows = label(formation)
            formation_pred = [
                np.max(np.where(formation_windows == island_id)) + track_start
                for island_id in np.unique(formation_windows)
                if island_id > 0
            ]

        if np.sum(breakdown) == 0:
            breakdown_pred = [-1]
        else:
            breakdown_windows = label(breakdown)
            breakdown_pred =  [
                np.min(np.where(breakdown_windows == island_id)) + track_start
                for island_id in np.unique(breakdown_windows)
                if island_id > 0
            ]

        return formation, breakdown

    def predict_step(self, batch, batch_idx):
        inputs = batch[self.hparams.x_key]

        #pad first and last time points so that each window is centered on its timepoint index
        inputs = torch.cat([inputs[:,0:1], inputs, inputs[:,-1:]], dim=1).unsqueeze(0)

        instance_logits= torch.stack([self(patch.squeeze(0)) for patch, _ in self.inference_splitter(inputs)]).squeeze(1) # B x 12

        window_logits = self.classifier_head(instance_logits) # B x 3

        instance_logits = rearrange(instance_logits,  'b (num_classes window_size ) -> b  num_classes window_size', num_classes = 4, window_size=self.hparams.window_size)

        instance_label = torch.argmax(instance_logits, dim=1)[1:-1].cpu().numpy() # remove padding
        window_label = torch.argmax(window_logits, dim=1)[1:-1].cpu().numpy() # remove padding
        stage='predict'


        # #save labeled movie of prediction
        # from PIL import Image, ImageDraw, ImageFont
        # font = ImageFont.load_default()
        # # Define text positions
        # x1, y1 = 0, 0
        # x2, y2 = 0, 20

        # movie = []
        # for il, wl, frame in zip(instance_label, window_label, batch[self.hparams.x_key].cpu().numpy().squeeze()):
        #     # add il and wl text to frame
        #     # Convert the NumPy array to a PIL Image
        #     image = Image.fromarray(frame.astype(float))

        #     # Create a drawing context
        #     draw = ImageDraw.Draw(image)
        #     # Draw text on the image
        #     draw.text((x1, y1), f'i:{il[0]}', fill='white', font=font)
        #     draw.text((x2, y2), f'w:{wl}', fill='white', font=font)

        #     # Convert the image back to a NumPy array
        #     movie.append(np.array(image))

        # OmeTiffWriter.save(uri = Path(self.hparams.save_dir) / f"{stage}_images" / f"{batch['track_id'].cpu().item()}.ome.tiff", data = np.stack(movie))

        track_start = batch['track_start'].cpu().item()
        track_id = batch['track_id'].cpu().item()
        formation= batch['formation'].cpu().item()
        breakdown= batch['breakdown'].cpu().item()

        formation_pred, breakdown_pred = self.instance_consensus(instance_label, track_start)

        # if np.sum( window_label == 1) == 0:
        #     formation_pred = [-1]
        # else:
        #     formation_windows = label(window_label == 1)
        #     formation_pred = [
        #         np.max(np.where(formation_windows == island_id)) + track_start
        #         for island_id in np.unique(formation_windows)
        #         if island_id > 0
        #     ]

        # if np.sum(window_label == 2) == 0:
        #     breakdown_pred = [-1]
        # else:
        #     breakdown_windows = label(window_label == 2)
        #     breakdown_pred =  [
        #         np.min(np.where(breakdown_windows == island_id)) + track_start
        #         for island_id in np.unique(breakdown_windows)
        #         if island_id > 0
        #     ]


        self.predictions  = pd.concat([self.predictions, pd.DataFrame({'track_id': [track_id], 'predicted_formation': [formation_pred], 'predicted_breakdown': [breakdown_pred], 'formation':[formation], 'breakdown':[breakdown]})])
        self.predictions.to_csv(Path(self.hparams.save_dir) / 'predictions.csv')
        np.save(Path(self.hparams.save_dir) / f'{track_id}_window.npy', window_label)
        np.save(Path(self.hparams.save_dir) / f'{track_id}_instance.npy', instance_label) 

        return window_logits 
