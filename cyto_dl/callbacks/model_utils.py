import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
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
        id_label: Optional[str] = None,
    ):
        """
        Args:
            x_label: x_label from datamodule
            id_field: id_field from datamodule
        """
        super().__init__()

        self.x_label = x_label
        self.id_label = id_label
        self.cutoff_kld_per_dim = 0

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        with torch.no_grad():
            embeddings = get_all_embeddings(
                trainer.datamodule.train_dataloader(),
                trainer.datamodule.val_dataloader(),
                trainer.datamodule.test_dataloader(),
                pl_module,
                self.x_label,
                self.id_label,
            )

            with tempfile.TemporaryDirectory() as tmp_dir:
                dest_path = os.path.join(tmp_dir, "embeddings.csv")
                embeddings.to_csv(dest_path)

                mlflow.log_artifact(local_path=dest_path, artifact_path="dataframes")


def get_all_embeddings(
    train_dataloader,
    val_dataloader,
    test_dataloader,
    pl_module: LightningModule,
    x_label: str,
    id_label: None,
):
    all_embeddings = []
    cell_ids = []
    split = []

    zip_iter = zip(["train", "val", "test"], [train_dataloader, val_dataloader, test_dataloader])

    with torch.no_grad():
        for split_name, dataloader in zip_iter:
            log.info(f"Getting embeddings for split: {split_name}")

            _bs = dataloader.batch_size
            _len = len(dataloader) * dataloader.batch_size

            _embeddings = np.zeros((_len, pl_module.latent_dim))
            _split = np.empty(_len, dtype=object)
            _ids = None

            id_label = pl_module.hparams.get("id_label", None) if id_label is None else id_label

            for index, batch in enumerate(dataloader):
                if _ids is None:
                    if id_label is not None and id_label in batch:
                        _ids = np.empty(_len, dtype=batch[id_label].cpu().numpy().dtype)
                    else:
                        _ids = None

                for key in batch.keys():
                    if not isinstance(batch[key], list):
                        batch[key] = batch[key].to(pl_module.device)

                z_parts_params, z_composed = pl_module(batch, decode=False, compute_loss=False)

                mu_vars = z_parts_params[x_label]
                if mu_vars.shape[1] != pl_module.latent_dim:
                    mus = mu_vars[:, : int(mu_vars.shape[1] / 2)]
                else:
                    mus = mu_vars

                start = _bs * index
                end = start + len(mus)

                _embeddings[start:end] = mus.cpu().numpy()

                if _ids is not None:
                    _ids[start:end] = batch[id_label].detach().cpu().squeeze()
                _split[start:end] = [split_name] * len(mus)

            diff = _bs - len(batch)
            if diff > 0:
                # if last batch is smaller discard the difference
                _embeddings = _embeddings[:-diff]
                if _ids is not None:
                    _ids = _ids[:-diff]
                _split = _split[:-diff]
            all_embeddings.append(_embeddings)
            cell_ids.append(_ids)
            split.append(_split)

    all_embeddings = np.vstack(all_embeddings)
    cell_ids = np.hstack(cell_ids) if cell_ids[0] is not None else None
    split = np.hstack(split)

    df = pd.DataFrame(all_embeddings, columns=[f"mu_{i}" for i in range(all_embeddings.shape[1])])
    df["split"] = split
    if cell_ids is not None:
        df["CellId"] = cell_ids

    return df
