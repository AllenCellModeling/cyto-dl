from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MeanMetric

from cyto_dl.models.base_model import BaseModel

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from bioio.writers import OmeTiffWriter

class ChannelContrastive(BaseModel):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        task_head: nn.Module,
        x_key: str,
        save_dir: str = "./",
        **base_kwargs,
    ):
        """
        Parameters
        ----------
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

    def forward(self, x):
        return self.backbone(x)
    

    def plot_classes(self, embeddings):
        # calculate pca on predictions and label by labels
        pca = PCA(n_components=2)
        pca.fit(torch.cat(embeddings))

        # num_channels x batch x embedding dim
        embeddings = torch.stack(embeddings)
        
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'yellow', 'pink', 'brown', 'gray']
        shapes = ['o', 'x', 's', 'v', '^', '<', '>', 'd', 'p', 'h']
        idx = torch.randperm(embeddings.shape[1])[:10]
        for ch in range(embeddings.shape[0]):
            for i in range(10):
                ax.scatter(embeddings[ch, idx[i], 0], embeddings[ch, idx[i], 1], c=colors[i], marker=shapes[ch])

        fig.savefig(Path(self.hparams.save_dir) / f"{self.current_epoch}_pca.png")
        plt.close(fig)
  
    def model_step(self, stage, batch, batch_idx):
        x = batch[self.hparams.x_key].as_tensor()
        b, c, _, _, _= x.shape

        embeddings = [self.forward(x[:, ch].unsqueeze(1)) for ch in range(c)]

        all_channel_pairs = list(combinations(range(c), 2))
        loss = 0
        for p in all_channel_pairs:
            loss += self.task_head.run_head((embeddings[p[0]], embeddings[p[1]]), None, 'train')['loss']

        loss= loss / len(all_channel_pairs)

        if stage == 'val' and batch_idx == 0:
            with torch.no_grad():
                self.plot_classes([e.detach().cpu().float() for e in embeddings])

        return loss, None, None
    
    def predict_step(self, batch, batch_idx):
        x = batch[self.hparams.x_key].as_tensor()
        b, c, _, _, _= x.shape

        preds = []
        for ch in range(c):
            preds.append(self.backbone(x[:, ch].unsqueeze(1)).detach().cpu().float().numpy())
        preds = np.concatenate(preds, axis=0)

        preds = pd.DataFrame(preds, columns=[str(i) for i in range(preds.shape[1])])
        preds['channel'] = [ch for ch in range(c) for _ in range(b)]
        preds['batch_idx'] = batch_idx
        preds['structure'] = [batch['structure_name'][i] for _ in range(c) for i in range(b)]
        preds['crop_number'] = [i   for _ in range(c) for i in range(b)]
        preds.to_csv(Path(self.hparams.save_dir) / f"{batch_idx}_predictions.csv")
        OmeTiffWriter.save(uri = Path(self.hparams.save_dir) / f"{batch_idx}_predictions.tiff", data = x.detach().cpu().float().numpy())
        return None, None, None


