from pathlib import Path
from typing import Callable

import numpy as np
import torch
from aicsimageio.writers import OmeTiffWriter
from einops import reduce
from lightning.pytorch.callbacks import Callback
from monai.inferers import sliding_window_inference
from ostats import add_sample
from scipy.spatial.distance import mahalanobis


class OutlierDetection(Callback):
    def __init__(self, n_epochs, layer_names, save_dir, x_key: str = "raw"):
        super().__init__()
        self.n_epochs = n_epochs
        self.layer_names = layer_names
        self.save_dir = Path(save_dir)
        self.x_key = x_key

        self.default_dict = {ln: None for ln in layer_names}

        self.cov = self.default_dict.copy()
        self.mu = self.default_dict.copy()
        self.n = 0

        self.inverse_cov = {}
        self.mahalanobis_distances = {ln: [] for ln in layer_names}
        self.activations = {ln: [] for ln in layer_names}

        self._train_hook_registered = False

        self.out_distances = {}
        self.out_activations = {}
        self.patch_count = -1

    # State saving functions
    @property
    def state_key(self) -> str:
        return "OutlierDetection"

    def state_dict(self):
        return {"md_cov": self.cov, "md_mu": self.mu, "md_n": self.n}

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["callbacks"][self.state_key] = self.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        print("LOAD CKPT")
        # load covariance matrix and mean if present, otherwise initialize with defaults
        od_params = checkpoint.get(self.state_key, {})
        self.cov = od_params.get("md_cov", self.default_dict.copy())
        self.mu = od_params.get("md_mu", self.default_dict.copy())
        self.n = od_params.get("md_n", 0)
        # super().on_load_checkpoint(trainer, pl_module, checkpoint)

    def load_state_dict(self, state_dict):
        print("LOAD CKPT")
        # load covariance matrix and mean if present, otherwise initialize with defaults
        od_params = state_dict.get(self.state_key, {})
        self.cov = od_params.get("md_cov", self.default_dict.copy())
        self.mu = od_params.get("md_mu", self.default_dict.copy())
        self.n = od_params.get("md_n", 0)
        # super().load_state_dict(state_dict)

    # training functions
    def flatten_activations(self, act):
        return reduce(act.clone().detach(), "b c z y x ->  b c", reduction="mean").cpu().numpy()

    def update_covariance_hook(self, layer_name: str) -> Callable:
        def fn(_, __, output):
            """record spatial mean and cov of channel activations per image in batch."""
            output = self.flatten_activations(output)
            self.mu[layer_name] = (
                self.mu[layer_name]
                if self.mu[layer_name] is not None
                else np.zeros(output.shape[1])
            )
            self.cov[layer_name] = (
                self.cov[layer_name]
                if self.cov[layer_name] is not None
                else np.zeros((output.shape[1], output.shape[1]))
            )
            for out in output:
                # online covariance estimation
                add_sample(self.n, out, self.mu[layer_name], cov=self.cov[layer_name])
                self.n += 1

        return fn

    def on_train_epoch_start(self, trainer, pl_module):
        """set forward hook."""
        if (
            trainer.current_epoch >= (trainer.max_epochs - self.n_epochs)
            and not self._train_hook_registered
        ):
            named_modules = dict([*pl_module.backbone.named_modules()])
            for layer_name in self.layer_names:
                named_modules[layer_name].register_forward_hook(
                    self.update_covariance_hook(layer_name)
                )
            self._train_hook_registered = True

    # inference functions
    def calculate_mahalanobis_hook(self, layer_name):
        def fn(_, __, output):
            output = self.flatten_activations(output)
            mean = self.mu[layer_name]
            vi = self.inverse_cov[layer_name]
            for out in output:
                md = mahalanobis(out, mean, vi)
                self.mahalanobis_distances[layer_name].append(md)
                self.activations[layer_name].append(out)
            assert False

        return fn

    def _inference_start(self, pl_module):
        """add mahalanobis calculation hook and calculate inverse covariance matrix."""
        if self.n > 0:
            # save sliding window args for mahalanobis image creation
            pl_module.hparams.inference_args["overlap"] = 0
            pl_module.hparams.inference_args["mode"] = "constant"

            self.inference_args = pl_module.hparams.inference_args

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

    # image creation functions
    def make_mahalanobis_image(self, x, img_name, distances):
        """line up mahalanobis distances with spatial locations and save as image."""
        template_image = self._get_template_image(x)
        img_name = "test"

        for layer_name, mahalanobis_distances in distances.items():
            layer_image = template_image.copy()
            for idx, md in enumerate(mahalanobis_distances):
                layer_image[layer_image == idx] = md
            OmeTiffWriter.save(
                uri=self.save_dir / f"{img_name}_{layer_name}_mahalanobis.tif",
                data=layer_image,
                dimension_order="ZYX",
            )

    def _simple_forward(self, x):
        """forward function that counts patch indices."""
        self.patch_count += 1
        return torch.ones_like(x) * self.patch_count

    def _get_template_image(self, x):
        """Generate an image where the value of each pixel is the index of that patch in the
        sliding window."""
        self.patch_count = -1
        template_image = sliding_window_inference(
            inputs=x.unsqueeze(0),
            predictor=self._simple_forward,
            device=torch.device("cpu"),
            **self.inference_args,
        )
        return template_image.int().numpy().astype(float)

    def _inference_batch_end(self, batch):
        # associate activations and mahalanobis distances with filenames
        if self.n > 0:
            batch_names = batch[f"{self.x_key}_meta_dict"]["filename_or_obj"]
            assert len(batch_names) == 1, "batch size must be 1 for Outlier Detection"
            # activations are saved per-patch
            distances_per_image = len(self.mahalanobis_distances[self.layer_names[0]]) // len(
                batch_names
            )
            for i, name in enumerate(batch_names):
                image_distances = {
                    k: v[i * distances_per_image : (i + 1) * distances_per_image]
                    for k, v in self.mahalanobis_distances.items()
                }
                self.out_distances[name] = image_distances

                self.out_activations[name] = {
                    k: v[i * distances_per_image : (i + 1) * distances_per_image]
                    for k, v in self.activations.items()
                }

                self.make_mahalanobis_image(batch[self.x_key][i], name, image_distances)

            # reset between batches
            self.mahalanobis_distances = {ln: [] for ln in self.layer_names}
            self.activations = {ln: [] for ln in self.layer_names}

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._inference_batch_end(batch)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._inference_batch_end(batch)

    def _inference_epoch_end(self):
        # save out results
        np.save(self.save_dir / "mahalanobis.npy", self.out_distances)
        np.save(self.save_dir / "activations.npy", self.out_activations)

    def on_test_epoch_end(self, trainer, pl_module):
        self._inference_epoch_end()

    def on_predict_epoch_end(self, trainer, pl_module):
        self._inference_epoch_end()
