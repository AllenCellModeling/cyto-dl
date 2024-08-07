from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torchmetrics import MeanMetric

from cyto_dl.models.base_model import BaseModel


class Contrastive(BaseModel):
    def __init__(
        self,
        backbone: nn.Module,
        task_head: nn.Module,
        anchor_key: str = "image",
        positive_key: str = "image_aug",
        target_key: str = "target",
        meta_keys: list[str] = [],
        save_dir: str = "./",
        viz_freq: int = 10,
        **base_kwargs,
    ):
        """
        Parameters
        ----------
        backbone: nn.Module
            Backbone model
        task_head: nn.Module
            Task head model
        anchor_key: str
            Key in batch dictionary for anchor image
        positive_key: str
            Key in batch dictionary for positive image
        target_key: str
            OPTIONAL Key in batch dictionary for target, used only for visualization
        meta_keys: list[str]
            List of keys in batch dictionary to save to csv during prediction
        save_dir: str
            Directory to save visualizations
        viz_freq: int
            Frequency to save visualizations
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

        # plot PC1 vs PC2 as heatmap
        embedding1 = pca.transform(embedding1)
        fig, ax = plt.subplots()
        counts, xedges, yedges = np.histogram2d(embedding1[:, 0], embedding1[:, 1], bins=30)
        ax.imshow(counts, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower")
        fig.savefig(Path(self.hparams.save_dir) / f"{self.current_epoch}_heatmap.png")
        plt.close(fig)

        # Plot anchor/positive relationship for a subsample
        random_examples = np.random.choice(embedding1.shape[0], 10)
        embedding1 = embedding1[random_examples]
        embedding2 = pca.transform(embedding2[random_examples])

        fig, ax = plt.subplots()
        # plot anchor embeddings in gray
        ax.scatter(embedding1[:, 0], embedding1[:, 1], c="green")

        # plot positive embeddings in green
        ax.scatter(embedding2[:, 0], embedding2[:, 1], c="green")

        # draw lines between anchor and positive, anchor and negative
        ax.plot([embedding1[:, 0], embedding2[:, 0]], [embedding1[:, 1], embedding2[:, 1]], "gray")

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
        x1 = batch[self.hparams.anchor_key].as_tensor()
        x2 = batch[self.hparams.positive_key].as_tensor()

        backbone_features = self.forward(x1, x2)
        out = self.task_head.run_head(backbone_features, batch, stage)

        if stage == "val" and batch_idx == 0:
            with torch.no_grad():
                embedding1 = out["y_hat_out"].detach().cpu().numpy()
                if self.hparams.target_key in batch:
                    labels = batch[self.hparams.target_key]
                    if isinstance(labels, torch.Tensor):
                        labels = labels.cpu().numpy()
                    self.plot_classes(embedding1, labels)
                else:
                    embedding2 = out["y_out"].detach().cpu().numpy()
                    self.plot_neighbors(embedding1, embedding2)

        return out["loss"], None, None

    def predict_step(self, batch, batch_idx):
        x = batch[self.hparams.anchor_key]
        embeddings = self.backbone(x if isinstance(x, torch.Tensor) else x.as_tensor())
        return embeddings.detach().cpu().numpy(), x.meta
