from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional
from aics_im2im.datamodules.shapenet_dataset import (
    Shapes3dDataset,
    IndexField,
)
from aics_im2im.datamodules.shapenet_dataset.utils import (
    get_data_fields,
    get_inputs_field,
)


class ShapenetDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_folder: str = "/allen/aics/modeling/ritvik/projects/occupancy_networks/data/ShapeNet",
        method: str = "shapenet_dfnet",
        dataset_type: str = "partial_pointcloud",
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        points_subsample: int = 2048,
        input_type: str = "pointcloud",
        points_file: str = "points.npz",
        points_iou_file: str = "points.npz",
        pointcloud_n: int = 2048,
        pointcloud_noise: float = 0.005,
        pointcloud_file: str = "pointcloud.npz",
        part_ratio: float = 0.5,
        partial_type: str = "centery_random",
        return_idx: bool = True,
        train_rotate: bool = True,
        train_translate: bool = False,
        train_single_trans: bool = False,
        batch_size: int = 10,
        num_workers: int = 0,
        multi_files: Optional[int] = None,
        categories: Optional[str] = None,
    ):
        """ """
        super().__init__()
        self.dataset_folder = dataset_folder
        self.method = method
        self.dataset_type = dataset_type
        self.dataset_folder = dataset_folder
        self.categories = categories
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.points_subsample = points_subsample
        self.input_type = input_type
        self.points_file = points_file
        self.multi_files = multi_files
        self.points_iou_file = points_iou_file
        self.pointcloud_n = pointcloud_n
        self.pointcloud_noise = pointcloud_noise
        self.pointcloud_file = pointcloud_file
        self.part_ratio = part_ratio
        self.partial_type = partial_type
        self.return_idx = return_idx
        self.train_rotate = train_rotate
        self.train_translate = train_translate
        self.train_single_trans = train_single_trans
        self.categories = categories
        self.loader_fnc = DataLoader
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.splits = {"train": train_split, "val": val_split, "test": test_split}

    def _get_dataset(self, mode):
        fields = get_data_fields(
            mode,
            self.points_subsample,
            self.input_type,
            self.points_file,
            self.multi_files,
            self.points_iou_file,
        )

        inputs_field = get_inputs_field(
            mode,
            self.input_type,
            self.pointcloud_n,
            self.pointcloud_noise,
            self.pointcloud_file,
            self.multi_files,
            self.part_ratio,
            self.partial_type,
        )

        if inputs_field is not None:
            fields["inputs"] = inputs_field

        if self.return_idx:
            fields["idx"] = IndexField()
        transform = []
        if mode == "train":
            if self.train_rotate:
                transform.append("rotate")
            if self.train_translate:
                transform.append("translate")
            if self.train_single_trans:
                transform.append("single_trans")

        dataset = Shapes3dDataset(
            self.dataset_folder,
            fields,
            self.splits[mode],
            self.categories,
            transform,
        )
        return dataset

    def train_dataloader(self):
        mode = "train"
        self.train_dataset = self._get_dataset(mode)
        dataloader = self.loader_fnc(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self):
        mode = "val"
        self.val_dataset = self._get_dataset(mode)
        dataloader = self.loader_fnc(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        mode = "test"
        self.test_dataset = self._get_dataset(mode)
        dataloader = self.loader_fnc(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader
