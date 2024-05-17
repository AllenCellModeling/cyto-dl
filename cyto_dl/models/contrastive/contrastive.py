from pathlib import Path

import numpy as np
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
        x_key: str,
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
        x_key: str
            key of input image in batch
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
        
        random_examples = np.random.choice(embedding1.shape[0], 10)
        embedding1 = pca.transform(embedding1[random_examples])
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

        # plot pca
        fig, ax = plt.subplots()
        scatter = ax.scatter(pca_predictions[:, 0], pca_predictions[:, 1], c=labels)
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        fig.savefig(Path(self.hparams.save_dir) / f"{self.current_epoch}_classes.png")
        plt.close(fig)
  
    def model_step(self, stage, batch, batch_idx):
        x1= batch['image'].as_tensor()
        x2 = batch['image_aug'].as_tensor()

        backbone_features = self.forward(x1, x2)
        out = self.task_head.run_head(backbone_features, batch, stage)

        if stage == 'val' and batch_idx == 0:
            with torch.no_grad():
                embedding1 = out['y_hat_out'].detach().cpu().numpy()
                embedding2 = out['y_out'].detach().cpu().numpy()
                if self.hparams.target_key in batch:
                    self.plot_classes(embedding1, batch[self.hparams.target_key].detach().cpu().numpy())
                else:
                    self.plot_neighbors(embedding1, embedding2)

        return out['loss'], None, None
