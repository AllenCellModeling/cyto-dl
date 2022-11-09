from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from monai.data import DataLoader, SmartCacheDataset, Dataset


class PatchDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        manifest_path: str,
        dataloader_kwargs: dict = None,
        buffer_size: int = 30,
        buffer_replace_rate=1.0,
        transforms: list = [],
        **kwargs
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.transforms = transforms
        self.buffer_size = buffer_size
        self.replace_rate = buffer_replace_rate

    def make_manifest_dataset(self, manifest: str, transform, n_imgs=-1, test=False):
        dataframe = pd.read_csv(manifest)
        data_list = [
            {col: dataframe[col].iloc[i] for col in dataframe.columns}
            for i in range(len(dataframe))
        ]
        if n_imgs < 0:
            n_imgs = len(data_list)
        if not test:
            return SmartCacheDataset(
                data=data_list,
                transform=transform,
                replace_rate=self.replace_rate,
                cache_num=n_imgs,
            )
        else:
            return Dataset(data=data_list, transform=transform)

    def make_dataloader(self, dataset):
        return DataLoader(dataset=dataset, **self.dataloader_kwargs)

    def train_dataloader(self):
        dataset = self.make_manifest_dataset(
            self.manifest_path / "train.csv",
            self.transforms["train"],
            self.buffer_size["train"],
        )
        return self.make_dataloader(dataset)

    def val_dataloader(self):
        dataset = self.make_manifest_dataset(
            self.manifest_path / "valid.csv",
            self.transforms["valid"],
            self.buffer_size["valid"],
        )
        return self.make_dataloader(dataset)

    def _make_test_dataloader(self):
        dataset = self.make_manifest_dataset(
            self.manifest_path / "test.csv", self.transforms["test"], test=True
        )
        return self.make_dataloader(dataset)

    def test_dataloader(self):
        return self._make_test_dataloader()

    def predict_dataloader(self):
        return self._make_test_dataloader()
