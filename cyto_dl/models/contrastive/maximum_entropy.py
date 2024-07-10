import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchmetrics import MeanMetric

from cyto_dl.models.base_model import BaseModel
from cyto_dl.nn.maximum_entropy import MECLoss

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



class MaximumEntropy(BaseModel):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        n_iter_per_epoch: int,
        eps_d: float = 64,
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
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
        }
        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.model = model
        self.loss_fn = MECLoss
        # mu is only used for MEC metric
        d = self.model.dim
        eps_d /= d
        print(f"eps_d: {eps_d}")
        lamda = 1/(batch_size * eps_d)

        self.lambda_scheduler = self.lamda_scheduler(8/lamda, 1/lamda, epochs, n_iter_per_epoch, warmup_epochs=10)
        self.momentum_scheduler = self.cosine_scheduler(0.996, 1, epochs, n_iter_per_epoch)
        self.automatic_optimization = False

    def cosine_scheduler(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule


    def lamda_scheduler(self, start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule


    def forward(self, x1, x2):
        return self.model(x1, x2)

    def plot_classes(self, predictions, labels):
        # calculate pca on predictions and label by labels
        pca = PCA(n_components=2)
        predictions= predictions.detach().cpu().numpy()
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

    def model_step(self, stage, batch, batch_idx):
        if 'label' not in batch:
            z1, z2, p1, p2 = self.forward(batch["image"], batch["image_aug"])
            lambda_inv = self.lambda_scheduler[self.global_step]
            momentum = self.momentum_scheduler[self.global_step]
            mec_loss = (self.loss_fn(p1, z2, lambda_inv) + self.loss_fn(p2, z1, lambda_inv)) * 0.5 / self.hparams.batch_size
            loss = -1 * mec_loss * lambda_inv
            if stage == 'train' and batch_idx==0 and  (self.current_epoch + 1) % self.hparams.save_every_n_epochs == 0:
                self.plot_classes(z1, batch["target"])
        else:
            class_pred= self.forward(batch['image'])
            loss = self.loss_fn(class_pred, batch['label'])
            # calculate accuracy
            pred = torch.argmax(class_pred, dim=1)
            acc = torch.sum(pred == batch['label']).item() / len(pred)
            self.log(f"{stage}/acc", acc)

        if stage == 'train':
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

            # momentum update of the parameters of the teacher network
            with torch.no_grad():
                for param_q, param_k in zip(self.model.encoder.parameters(), self.model.teacher.parameters()):
                    param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)



        return loss, None, None
