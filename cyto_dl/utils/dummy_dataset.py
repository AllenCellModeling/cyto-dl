import torch
from lightning import LightningDataModule
from monai.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset


class DummyDatamodule(LightningDataModule):
    def __init__(self, num_samples, batch_size, shapes, dummy_metadata=None, **kwargs):
        super().__init__()
        self.shapes = shapes
        self.num_samples = num_samples
        self.batch_size = batch_size
        if dummy_metadata is not None:
            if isinstance(dummy_metadata, DictConfig):
                dummy_metadata = OmegaConf.to_container(dummy_metadata)
        self.dummy_metadata = dummy_metadata

    def get_dataloader(self):
        return DataLoader(
            DummyDataset(self.num_samples, self.dummy_metadata, **self.shapes),
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return self.get_dataloader()

    def val_dataloader(self):
        return self.get_dataloader()

    def test_dataloader(self):
        return self.get_dataloader()

    def predict_dataloader(self):
        return self.get_dataloader()


class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 10000, dummy_metadata: dict = None, **shapes):
        """
        Args:
            num_samples: how many samples to use in this dataset
            metadata: whether to generate metatensors
            **shapes: kwargs, where each key will become a batch key and each
                      value is the shape of the corresponding batch elements
        """
        super().__init__()
        self.shapes = shapes
        self.dummy_metadata = dummy_metadata

        if num_samples < 1:
            raise ValueError("Provide an argument greater than 0 for `num_samples`")

        self.num_samples = num_samples

    def generate_img(self, k):
        if "seg" in k:
            im = torch.zeros(*self.shapes[k])
            slicee = [slice(s // 2 - s // 4, s // 2 + s // 4, None) for s in self.shapes[k]]
            im[slicee] = 1
            return im
        return torch.randn(*self.shapes[k])

    def __getitem__(self, idx: int):
        out = {k: self.generate_img(k) for k in self.shapes.keys()}

        if self.dummy_metadata is not None:
            out.update(self.dummy_metadata)

        return out

    def __len__(self):
        return self.num_samples
