import csv
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from einops import reduce
from lightning.pytorch.callbacks import Callback
from online_stats import add_sample
from scipy.spatial.distance import mahalanobis


class OutlierDetection(Callback):
    def __init__(self, n_epochs, layer_names, save_dir):
        super().__init__()
        self.n_epochs = n_epochs
        self.layer_names = layer_names
        self.save_dir = Path(save_dir)

        self.cov = {ln: None for ln in layer_names}
        self.mu = {ln: None for ln in layer_names}
        self.inverse_cov = {}
        self.mahalanobis_distances = {ln: [] for ln in layer_names}
        self.activations = {ln: [] for ln in layer_names}
        self.n = 0
        self._run = False

        # initialize csvs
        init_mahalanobis_csv = {"file": []}
        for ln in layer_names:
            for metric in ("min", "max", "mean", "median", "var"):
                init_mahalanobis_csv[f"{ln}_mahalanobis_{metric}"] = []
        pd.DataFrame(init_mahalanobis_csv).to_csv(self.save_dir / "mahalanobis.csv", index=False)

        init_activation_csv = {"file": []}
        for ln in layer_names:
            init_activation_csv[f"{ln}_activation"] = []
        pd.DataFrame(init_activation_csv).to_csv(self.save_dir / "activations.csv", index=False)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["outlier_detection"] = {
            "md_cov": self.cov,
            "md_mu": self.mu,
            "md_n": self.n,
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        od_params = checkpoint.get("outlier_detection", {})
        self.cov = od_params.get("md_cov", None)
        self.mu = od_params.get("md_mu", None)
        self.n = od_params.get("md_n", None)
        if self.cov is None or self.mu is None:
            print("No mean or covariance matrices found, outlier detection skipped...")
            return
        self._run = True

    def flatten_activations(self, act):
        return reduce(act.clone().detach(), "b c z y x ->  b c", reduction="mean").cpu().numpy()

    def update_covariance_hook(self, layer_name: str) -> Callable:
        def fn(_, __, output):
            self._update_covariance(output, layer_name)

        return fn

    def _update_covariance(self, output, layer_name):
        """Record spatial mean and cov of channel activations per image in batch."""
        output = self.flatten_activations(output)
        if self.mu[layer_name] is None:
            self.mu[layer_name] = np.zeros(output.shape[1])
        if self.cov[layer_name] is None:
            self.cov[layer_name] = np.zeros((output.shape[1], output.shape[1]))
        for out in output:
            # online covariance estimation
            add_sample(self.n, out, self.mu[layer_name], cov=self.cov[layer_name])
            self.n += 1

    def on_train_epoch_start(self, trainer, pl_module):
        """Set forward hook."""
        if trainer.current_epoch == trainer.max_epochs - self.n_epochs:
            named_modules = dict([*pl_module.backbone.named_modules()])
            for layer_name in self.layer_names:
                named_modules[layer_name].register_forward_hook(
                    self.update_covariance_hook(layer_name)
                )
            self._run = True

    def calculate_mahalanobis_hook(self, layer_name):
        def fn(_, __, output):
            self._calculate_mahalanobis(output, layer_name)

        return fn

    def _calculate_mahalanobis(self, output, layer_name):
        output = self.flatten_activations(output)
        mean = self.mu[layer_name]
        vi = self.inverse_cov[layer_name]
        for out in output:
            md = mahalanobis(out, mean, vi)
            self.mahalanobis_distances[layer_name].append(md)
            self.activations[layer_name].append(out)

    def _inference_start(self, pl_module):
        """Add mahalanobis calculation hook and calculate inverse covariance matrix."""
        if self._run:
            named_modules = dict([*pl_module.backbone.named_modules()])
            for layer_name in self.layer_names:
                named_modules[layer_name].register_forward_hook(
                    self.calculate_mahalanobis_hook(layer_name)
                )
                self.inverse_cov[layer_name] = np.linalg.inv(self.cov[layer_name])

    def on_test_epoch_start(self, trainer, pl_module):
        self._inference_start(pl_module)

    def on_predict_epoch_start(self, trainer, pl_module):
        self._inference_start(pl_module)

    def _inference_batch_end(self, batch):
        if self._run:
            batch_names = batch["raw"].meta["filename_or_obj"]
            # activations are saved per-patch
            distances_per_image = len(self.mahalanobis_distances[self.layer_names[0]]) // len(
                batch_names
            )
            with open(self.save_dir / "mahalanobis.csv", "a", newline="") as file:
                writer = csv.writer(file, delimiter=",")
                rows = []
                for i, name in enumerate(batch_names):
                    summary = [name]
                    for k, v in self.mahalanobis_distances.items():
                        # compile stats per-image
                        md = v[i * distances_per_image : (i + 1) * distances_per_image]
                        summary += [
                            np.nanmin(md),
                            np.nanmax(md),
                            np.nanmean(md),
                            np.nanmedian(md),
                            np.nanvar(md),
                        ]
                    rows.append(summary)
                writer.writerows(rows)

            with open(self.save_dir / "activations.csv", "a", newline="") as file:
                writer = csv.writer(file, delimiter=",")
                rows = []
                for i, name in enumerate(batch_names):
                    for ln in self.layer_names:
                        # compile stats per-image
                        summary = [name] + [np.stack(self.activations[ln]).mean(0).tolist()]
                        rows.append(summary)
                writer.writerows(rows)

            self.mahalanobis_distances = {ln: [] for ln in self.layer_names}
            self.activations = {ln: [] for ln in self.layer_names}

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._inference_batch_end(batch)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._inference_batch_end(batch)
