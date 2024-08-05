import hydra
import numpy as np
from omegaconf import OmegaConf


def create_dataloader(data_cfg, data=None):
    """Create a dataloader from a data config and optional data."""
    data_cfg = OmegaConf.to_object(data_cfg)
    if data is not None:
        # inference, using make_array_dataloader
        if "data" in data_cfg:
            data_cfg["data"] = data
        # training, has train_dataloaders/val_dataloaders
        for split in ("train", "val", "test"):
            if f"{split}_dataloaders" in data_cfg:
                data_cfg[f"{split}_dataloaders"]["data"] = data[split]

    # Instantiate the dataloader with the dataset
    dataloader = hydra.utils.instantiate(data_cfg)

    return dataloader


def extract_array_predictions(output, task_heads=None):
    """Converts output from model.predict() to a dictionary of numpy arrays per head."""
    predictions = {}
    for batch_pred in output:
        # ignore io_map
        _, batch_pred = batch_pred
        # if no task_heads are provided, use all
        if task_heads is None:
            task_heads = list(batch_pred.keys())
        # combine all predictions per-head
        for head in task_heads:
            if head not in predictions:
                predictions[head] = []
            predictions[head] += batch_pred[head]["pred"]
    # stack head predictions into numpy array
    for head, pred in predictions.items():
        predictions[head] = np.stack(pred)

    return predictions
