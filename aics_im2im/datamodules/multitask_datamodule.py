from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai.data import SmartCacheDataset


class PatchDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        images_per_epoch: int,
        manifest_path: str,
        dataloader_kwargs: dict = None,
        transforms: list = [],
    ):
        super().__init__()
        manifest_path = Path(manifest_path)
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.images_per_epoch = images_per_epoch

        self.train = self.make_manifest_dataset(
            manifest_path / "train.csv", transforms["train"]
        )
        self.valid = self.make_manifest_dataset(
            manifest_path / "valid.csv", transforms["valid"]
        )
        self.test = self.make_manifest_dataset(
            manifest_path / "test.csv", transforms["test"]
        )

    def make_manifest_dataset(self, manifest: str, transform):
        dataframe = pd.read_csv(manifest)
        dataframe = dataframe.sample(self.images_per_epoch)
        data_list = [
            {col: dataframe[col].iloc[i] for col in dataframe.columns}
            for i in range(len(dataframe))
        ]

        return SmartCacheDataset(data=data_list, transform=transform)

    def make_dataloader(self, dataset):
        return DataLoader(dataset=dataset, **self.dataloader_kwargs)

    def train_dataloader(self):
        return self.make_dataloader(self.train)

    def val_dataloader(self):
        return self.make_dataloader(self.valid)

    def test_dataloader(self):
        return self.make_dataloader(self.test)