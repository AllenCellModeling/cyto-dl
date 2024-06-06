from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchmetrics import MeanMetric

from cyto_dl.models.base_model import BaseModel

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Contrastive(BaseModel):
    def __init__(
        self,
        backbone: nn.Module,
        task_head: nn.Module,
        anchor_key: str = 'image',
        positive_key: str = 'image_aug',
        target_key: str = 'target',
        save_dir: str = "./",
        viz_freq: int = 10,
        *,
        model: nn.Module,
        **base_kwargs,
    ):
        """
        Parameters
        ----------
        model: nn.Module
            model network, parameters are shared between task heads
        save_dir="./"
            directory to save images during training and validation
        save_images_every_n_epochs=1
            Frequency to save out images during training
        **base_kwargs:
            Additional arguments passed to BaseModel
        """
        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
        }
        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.backbone = backbone
        self.task_head = task_head

    def forward(self, x1, x2):
        return self.backbone(x1), self.backbone(x2)

    def plot_neighbors(self, embedding1, embedding2):
        # calculate pca on predictions and label by labels
        pca = PCA(n_components=2)
        pca.fit(embedding1)
        
        embedding1 = pca.transform(embedding1)
        fig, ax = plt.subplots()
        counts, xedges, yedges = np.histogram2d(embedding1[:, 0], embedding1[:, 1], bins=30)
        ax.imshow(counts, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower')
        fig.savefig(Path(self.hparams.save_dir) / f"{self.current_epoch}_heatmap.png")
        plt.close(fig)
        
        random_examples = np.random.choice(embedding1.shape[0], 10)
        embedding1 = embedding1[random_examples]
        embedding2 = pca.transform(embedding2[random_examples])

        fig, ax = plt.subplots()

        # plot anchor embeddings in gray
        ax.scatter(embedding1[:, 0], embedding1[:, 1], c='green')

        # plot positive embeddings in green
        ax.scatter(embedding2[:, 0], embedding2[:, 1], c='green')


        # draw lines between anchor and positive, anchor and negative
        ax.plot([embedding1[:, 0], embedding2[:, 0]], [embedding1[:, 1], embedding2[:, 1]], 'gray')

        fig.savefig(Path(self.hparams.save_dir) / f"{self.current_epoch}_neighbors.png")
        plt.close(fig)

    
    def plot_classes(self, predictions, labels):
        # calculate pca on predictions and label by labels
        pca = PCA(n_components=2)
        pca.fit(predictions)
        pca_predictions = pca.transform(predictions)
        # convert labels to integers
        categories = list(np.unique(labels))
        labels = [categories.index(label) for label in labels]

        # plot pca
        fig, ax = plt.subplots()
        scatter = ax.scatter(pca_predictions[:, 0], pca_predictions[:, 1], c=labels)
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        fig.savefig(Path(self.hparams.save_dir) / f"{self.current_epoch}_classes.png")
        plt.close(fig)
  
    def model_step(self, stage, batch, batch_idx):
        x1= batch[self.hparams.anchor_key].as_tensor()
        x2 = batch[self.hparams.positive_key].as_tensor()

        backbone_features = self.forward(x1, x2)
        out = self.task_head.run_head(backbone_features, batch, stage)

        if stage == 'val' and batch_idx == 0:
            with torch.no_grad():
                embedding1 = out['y_hat_out'].detach().cpu().numpy()
                if self.hparams.target_key in batch:
                    self.plot_classes(embedding1, batch[self.hparams.target_key])
                else:
                    embedding2 = out['y_out'].detach().cpu().numpy()
                    self.plot_neighbors(embedding1, embedding2)

        return out['loss'], None, None

    def predict_step(self, batch, batch_idx):
        x = batch[self.hparams.anchor_key].as_tensor()
        embeddings = self.backbone(x)
        preds = pd.DataFrame(embeddings.detach().cpu().numpy(), columns=[str(i) for i in range(embeddings.shape[1])])
        preds['filename'] = batch['filename']
        preds['Gene'] = batch['Gene'] #.detach().cpu().numpy()
        preds['drug_label'] = batch['drug_label'] #.detach().cpu().numpy()
        preds.to_csv(Path(self.hparams.save_dir) / f"{batch_idx}_predictions.csv")
        return None, None, None