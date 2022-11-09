from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from monai.data import DataLoader, Dataset, PersistentDataset


class PatchDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        manifest_path: str,
        cache_dir: str,
        dataloader_kwargs: dict = None,
        buffer_size= {
            'train': -1, 'test':-1, 'valid':-1
        },
        transforms: list = [],
        **kwargs
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.transforms = transforms
        self.buffer_size = buffer_size
        self.cache_dir = cache_dir

    def make_manifest_dataset(self, manifest: str, transform, n_imgs=-1, test=False):
        dataframe = pd.read_csv(manifest)
        if n_imgs < 0:
            n_imgs = len(dataframe)
        dataframe = dataframe.sample(n_imgs)

        data_list = [
            {col: dataframe[col].iloc[i] for col in dataframe.columns}
            for i in range(len(dataframe))
        ]

        if not test:
            return PersistentDataset(
                data=data_list,
                transform=transform,
                cache_dir=self.cache_dir,
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
