import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from lightning import Callback, LightningModule, Trainer

log = logging.getLogger(__name__)


def save_predictions_classifier(preds, output_dir):
    """
    TODO: make this better? maybe use vol predictor code?
    TODO: drop unnecessary index
    """
    records = []
    for pred in preds:
        record = {}
        for col in ("id", "y", "yhat"):
            record[col] = pred[col].squeeze().numpy()
        record["loss"] = [pred["loss"].item()] * len(pred["id"])
        records.append(pd.DataFrame(record))

    pd.concat(records).reset_index().drop(columns="index").to_csv(
        Path(output_dir) / "model_predictions.csv", index_label=False
    )


class GetEmbeddings(Callback):
    """"""

    def __init__(
        self,
        x_label: str,
        loss: torch.nn.Module,
        latent_dim: int = 256,
        id_label: Optional[str] = None,
        sample_points: Optional[bool] = False,
        skew_scale: Optional[int] = 300,
        save_path: Optional[str] = None,
    ):
        """
        Args:
            x_label: x_label from datamodule
            id_field: id_field from datamodule
        """
        super().__init__()

        self.x_label = x_label
        self.id_label = id_label
        self.sample_points = sample_points
        self.skew_scale = skew_scale
        self.loss = loss
        self.save_path = save_path
        self.latent_dim = latent_dim

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        with torch.no_grad():
            embeddings = get_all_embeddings(
                trainer.datamodule.train_dataloader(),
                trainer.datamodule.val_dataloader(),
                trainer.datamodule.test_dataloader(),
                self.loss,
                pl_module,
                self.x_label,
                self.id_label,
                self.sample_points,
                self.skew_scale,
                self.latent_dim,
            )
            embeddings.to_csv(Path(self.save_path) / "embeddings.csv")


def gen_forward(pl_module, batch, x_label):
    if hasattr(pl_module, "backbone"):
        features, backward_indexes, patch_size = pl_module.backbone.encoder(
            batch[x_label]
        )
        predicted_img, mask = pl_module.backbone.decoder(
            features, backward_indexes, patch_size
        )
        xhat, z_parts_params = {}, {}
        xhat[x_label] = predicted_img
        z_parts_params[x_label] = features
    else:
        xhat, z_parts, z_parts_params = pl_module(
            batch, decode=True, inference=True, return_params=True
        )
    return xhat, z_parts_params


def get_all_embeddings(
    train_dataloader,
    val_dataloader,
    test_dataloader,
    this_loss,
    pl_module: LightningModule,
    x_label: str,
    id_label: None,
    sample_points: bool,
    skew_scale: int,
    latent_dim: int,
):
    all_embeddings = []
    cell_ids = []
    split = []
    all_loss = []

    zip_iter = zip(
        ["train", "val", "test"], [train_dataloader, val_dataloader, test_dataloader]
    )
    with torch.no_grad():
        for split_name, dataloader in zip_iter:
            log.info(f"Getting embeddings for split: {split_name}")

            _bs = dataloader.batch_size
            _len = len(dataloader) * dataloader.batch_size

            _embeddings = np.zeros((_len, latent_dim))
            _loss = np.zeros((_len, latent_dim))
            _split = np.empty(_len, dtype=object)
            _ids = None

            id_label = (
                pl_module.hparams.get("id_label", None)
                if id_label is None
                else id_label
            )

            for index, batch in enumerate(tqdm(dataloader)):
                if _ids is None:
                    if id_label is not None and id_label in batch:
                        _ids = np.empty(_len, dtype=np.array(batch[id_label]).dtype)
                    else:
                        _ids = None

                for key in batch.keys():
                    if not isinstance(batch[key], list):
                        batch[key] = batch[key].to(pl_module.device)
                import ipdb

                ipdb.set_trace()
                xhat, z_parts_params = gen_forward(pl_module, batch, x_label)
                # xhat, z_parts, z_parts_params = pl_module(
                #     batch, decode=True, inference=True, return_params=True
                # )

                if sample_points:
                    batch[x_label] = torch.tensor(
                        apply_sample_points(batch[x_label], True, skew_scale)
                    ).type_as(batch[x_label])
                    xhat[x_label] = torch.tensor(
                        apply_sample_points(xhat[x_label], True, skew_scale)
                    ).type_as(batch[x_label])

                if x_label in z_parts_params.keys():
                    mu_vars = z_parts_params[x_label]
                else:
                    mu_vars = z_parts_params["embedding"]

                mus = mu_vars
                if len(mu_vars.shape) > 2:
                    mus = mu_vars[:, 1:, :].mean(axis=1)

                rcl_per_input_dimension = this_loss(
                    xhat[x_label].contiguous(), batch[x_label].contiguous()
                )
                loss = (
                    rcl_per_input_dimension
                    # flatten
                    .view(rcl_per_input_dimension.shape[0], -1)
                    # and sum across each batch element's dimensions
                    .sum(dim=1)
                )
                loss = loss

                # if mu_vars.shape[1] != pl_module.latent_dim:
                #     mus = mu_vars[:, : int(mu_vars.shape[1] / 2)]
                # else:
                #     mus = mu_vars

                start = _bs * index
                end = start + len(mus)

                _embeddings[start:end] = mus.cpu().numpy()
                _loss[start:end] = loss.cpu().numpy()
                if _ids is not None:
                    _ids[start:end] = batch[id_label]
                _split[start:end] = [split_name] * len(mus)

            diff = _bs - len(batch)
            if diff > 0:
                # if last batch is smaller discard the difference
                _embeddings = _embeddings[:-diff]
                if _ids is not None:
                    _ids = _ids[:-diff]
                _split = _split[:-diff]
            all_embeddings.append(_embeddings)
            all_loss.append(_loss)
            cell_ids.append(_ids)
            split.append(_split)

    all_embeddings = np.vstack(all_embeddings)
    all_loss = np.vstack(all_loss)
    cell_ids = np.hstack(cell_ids) if cell_ids[0] is not None else None
    split = np.hstack(split)

    df = pd.DataFrame(
        all_embeddings, columns=[f"mu_{i}" for i in range(all_embeddings.shape[1])]
    )
    df["split"] = split
    if cell_ids is not None:
        df["CellId"] = cell_ids

    return df


def _sample(raw, skew_scale=100):
    num_points = 2048

    mask = torch.where(raw > 0, 1, 0).type_as(raw)

    disp = 0.001
    outs = torch.where(torch.ones_like(raw) > 0)
    if len(outs) == 3:
        z, y, x = outs
    else:
        y, x = outs

    probs = raw.clone()
    probs = probs.flatten()
    probs = probs / probs.max()

    skewness = skew_scale * (3 * (probs.mean() - torch.median(probs))) / probs.std()
    probs = torch.exp(skewness * probs)

    probs = torch.where(probs < 1e21, probs, 1e21)  # dont let sum of probs blow up

    # set probs to 0 outside mask
    inds = torch.where(mask.flatten() == 0)[0]
    probs[inds] = 0

    # scale probs so it sums to 1
    probs = probs / probs.sum()

    idxs = np.random.choice(
        np.arange(len(probs)),
        size=num_points,
        replace=True,
        p=probs.detach().cpu().numpy(),
    )
    x = x[idxs].detach().cpu() + 2 * (torch.rand(len(idxs)) - 0.5) * disp
    y = y[idxs].detach().cpu() + 2 * (torch.rand(len(idxs)) - 0.5) * disp
    if len(outs) == 3:
        z = z[idxs].detach().cpu() + 2 * (torch.rand(len(idxs)).type_as(x) - 0.5) * disp
    else:
        z = x.clone().detach().cpu()
        z.fill(0)
    new_cents = torch.stack([z, y, x], axis=1).float()
    assert new_cents.shape[0] == num_points
    return new_cents


def sample_points(orig, skew_scale):
    pcloud = []
    for i in range(orig.shape[0]):
        raw = orig[i, 0]
        new_cents = _sample(raw, skew_scale)
        pcloud.append(new_cents)
    pcloud = np.stack(pcloud, axis=0)
    return torch.tensor(pcloud)


def apply_sample_points(data, use_sample_points, skew_scale):
    if use_sample_points:
        return sample_points(data, skew_scale)
    else:
        return data
