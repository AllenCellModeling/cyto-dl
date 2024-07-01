from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional
from sklearn.decomposition import PCA
from torchmetrics import MeanMetric

from cyto_dl.models.base_model import BaseModel


class Triplet(BaseModel):
    def __init__(
        self,
        save_every_n_epochs: int = 100,
        save_dir: str = "./",
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
        compile: False
            Whether to compile the model using torch.compile
        **base_kwargs:
            Additional arguments passed to BaseModel
        """
        _DEFAULT_METRICS = {
            # "train/loss/mean_positive_dist": MeanMetric(),
            # "train/loss/closest_negative_dist": MeanMetric(),
            # "val/loss/mean_positive_dist": MeanMetric(),
            # "val/loss/closest_negative_dist": MeanMetric(),
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
        }
        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.model = model
        self.loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def plot_classes(self, predictions, labels):
        # calculate pca on predictions and label by labels
        pca = PCA(n_components=2)
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        pca.fit(predictions)
        pca_predictions = pca.transform(predictions)

        # plot pca
        fig, ax = plt.subplots()
        scatter = ax.scatter(pca_predictions[:, 0], pca_predictions[:, 1], c=labels)
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        fig.savefig(Path(self.hparams.save_dir) / f"{self.current_epoch}_pca.png")
        plt.close(fig)

    def find_hard_negatives(self, pairwise_dist, negatives_mask):
        # exlude positives
        pairwise_dist[~negatives_mask] = torch.inf
        hard_negative_idx = torch.argmin(pairwise_dist, dim=1)
        return hard_negative_idx

    def find_hard_positives(self, pairwise_dist, negatives_mask):
        # exclude negatives and self
        pairwise_dist[negatives_mask] = -torch.inf
        pairwise_dist[torch.eye(pairwise_dist.shape[0]).bool()] = -torch.inf

        hard_positive_idx = torch.argmax(pairwise_dist, dim=1)
        return hard_positive_idx

    def model_step(self, stage, batch, batch_idx):
        anchor_embeddings = self.model(batch["image"].squeeze(1))
        anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)

        # positive_embeddings = self.model(batch['image_aug'].squeeze(1))
        # positive_embeddings= torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)
        # find pairwisel2 distance between embeddings
        # pairwise_dist = torch.cdist(anchor_embeddings, positive_embeddings, p=2)

        pairwise_dist = torch.cdist(anchor_embeddings, anchor_embeddings, p=2)
        targets = batch["target"].unsqueeze(1).float()
        negatives_mask = torch.cdist(targets, targets, p=0).bool()

        hard_negative_idx = self.find_hard_negatives(pairwise_dist.clone(), negatives_mask)
        negative_embeddings = anchor_embeddings[hard_negative_idx]

        hard_positive_idx = self.find_hard_positives(pairwise_dist.clone(), negatives_mask)
        positive_embeddings = anchor_embeddings[hard_positive_idx]

        # # count how many hard negatives are used per label
        # hard_negative_counts = torch.bincount(batch["target"][hard_negative_idx])
        # print(hard_negative_counts)

        # reorder anchor embeddings to be matched as negative embeddings
        # negative_embeddings = anchor_embeddings[hard_negative_idx]

        # find triplet loss
        loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

        # with torch.no_grad():
        #     loss ={
        #         'loss': loss,
        #         'mean_positive_dist': torch.mean(pairwise_dist[~negatives_mask]).item(),
        #         'closest_negative_dist': torch.mean(torch.diagonal(pairwise_dist[hard_negative_idx])).item()
        #     }

        if stage == "val" and batch_idx == 0:
            print(
                "AVG HARD NEGATIVE DISTANCE:",
                pairwise_dist[torch.arange(pairwise_dist.shape[0]), hard_negative_idx].mean(),
            )
            print(
                "AVG HARD POSITIVE DISTANCE:",
                pairwise_dist[torch.arange(pairwise_dist.shape[0]), hard_positive_idx].mean(),
            )
            self.plot_classes(anchor_embeddings, batch["target"])
            # from aicsimageio.writers import OmeTiffWriter
            # OmeTiffWriter.save(uri=Path(self.hparams.save_dir) / f"{self.current_epoch}_anchors.tiff", data = batch['image'].squeeze().detach().cpu().numpy())
            # OmeTiffWriter.save(uri=Path(self.hparams.save_dir) / f"{self.current_epoch}_positives.tiff", data = batch['image_aug'].squeeze().detach().cpu().numpy())
            # OmeTiffWriter.save(uri=Path(self.hparams.save_dir) / f"{self.current_epoch}_negatives.tiff", data = batch['image'][max_indices].squeeze().detach().cpu().numpy())

        return loss, None, None
