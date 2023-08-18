import os
import logging
from torch.utils import data
import numpy as np
import yaml
import torch

logger = logging.getLogger(__name__)


# Fields
class Field(object):
    """Data fields class."""

    def load(self, data_path, idx, category):
        """Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        """
        raise NotImplementedError

    def check_complete(self, files):
        """Checks if set is complete.

        Args:
            files: files
        """
        raise NotImplementedError


class Shapes3dDataset(data.Dataset):
    """3D Shapes dataset class."""

    def __init__(
        self,
        dataset_folder,
        fields,
        split=None,
        categories=None,
        transform=None,
    ):
        """Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            transform (callable): transformation applied to data points
            cfg (yaml): config file
        """
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.transform = transform
        self.split = split
        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [
                c for c in categories if os.path.isdir(os.path.join(dataset_folder, c))
            ]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, "metadata.yaml")

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                self.metadata = yaml.load(f, Loader=yaml.Loader)
        else:
            self.metadata = {c: {"id": c, "name": "n/a"} for c in categories}

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]["idx"] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning("Category %s does not exist in dataset." % c)

            if split is None:
                self.models += [
                    {"category": c, "model": m}
                    for m in [
                        d
                        for d in os.listdir(subpath)
                        if (os.path.isdir(os.path.join(subpath, d)) and d != "")
                    ]
                ]

            else:
                split_file = os.path.join(subpath, split + ".lst")
                with open(split_file, "r") as f:
                    models_c = f.read().split("\n")

                if "" in models_c:
                    models_c.remove("")

                self.models += [{"category": c, "model": m} for m in models_c]

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.models)

    def __getitem__(self, idx):
        """Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        """
        category = self.models[idx]["category"]
        model = self.models[idx]["model"]
        c_idx = self.metadata[category]["idx"]
        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}

        info = c_idx

        for field_name, field in self.fields.items():
            field_data = field.load(model_path, idx, info, self.split)

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data["%s.%s" % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transforms(data, transform_type=self.transform)

        return data

    def transforms(self, data, transform_type=None):
        if "rotate" in transform_type:
            data["pcloud"], data["points"], data["points_iou"] = rotate_pointcloud(
                pointcloud=data["pcloud"],
                points=data["points"],
                points_iou=data.get("points_iou"),
            )

        if "translate" in transform_type:
            data["pcloud"], data["points"], data["points_iou"] = translate_pointcloud(
                pointcloud=data["pcloud"],
                points=data["points"],
                points_iou=data.get("points_iou"),
            )

        if "single_trans" in transform_type:
            points_df = None
            if "points.df" in data.keys():
                points_df = data["points.df"]
            points_iou_df = None
            if "points_iou.df" in data.keys():
                points_iou_df = data["points_iou.df"]

            if data.get("points_iou") is not None:
                (
                    data["pcloud"],
                    data["pcloud"],
                    data["points_iou"],
                    points_df,
                    points_iou_df,
                ) = single_translate_pointcloud(
                    pointcloud=data["pcloud"],
                    points=data["points"],
                    points_iou=data["points_iou"],
                    points_df=points_df,
                    points_iou_df=points_iou_df,
                )
                if points_df is not None:
                    data["points.df"] = points_df
                if points_iou_df is not None:
                    data["points_iou.df"] = points_iou_df
            else:
                data["pcloud"], data["points"], points_df = single_translate_pointcloud(
                    pointcloud=data["pcloud"],
                    points=data["points"],
                    points_df=points_df,
                )
                if points_df is not None:
                    data["points.df"] = points_df

        # clean None type keys
        filtered = {k: v for k, v in data.items() if v is not None}
        data.clear()
        data.update(filtered)

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        """Tests if model is complete.

        Args:
            model (str): modelname
        """
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s' % (field_name, model_path))
                return False

        return True


def collate_remove_none(batch):
    """Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    """
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    """Worker init function to ensure true randomness."""

    def set_num_threads(nt):
        try:
            import mkl

            mkl.set_num_threads(nt)
        except:
            pass
            torch.set_num_threads(1)
            os.environ["IPC_ENABLE"] = "1"
            for o in [
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            ]:
                os.environ[o] = str(nt)

    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)


def rotate_pointcloud(pointcloud, points, points_iou=None):
    theta = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(
        rotation_matrix
    )  # random rotation (x,z)
    points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)

    if points_iou is not None:
        points_iou[:, [0, 2]] = points_iou[:, [0, 2]].dot(rotation_matrix)

    return pointcloud, points, points_iou


def translate_pointcloud(pointcloud, points, points_iou=None):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype("float32")

    points = np.add(np.multiply(points, xyz1), xyz2).astype("float32")

    if points_iou is not None:
        points_iou = np.add(np.multiply(points_iou, xyz1), xyz2).astype("float32")

    return pointcloud, points, points_iou


def single_translate_pointcloud(
    pointcloud, points, points_iou=None, points_df=None, points_iou_df=None
):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[1])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )

    translated_points = np.add(np.multiply(points, xyz1), xyz2).astype("float32")

    translated_points_df = None
    if points_df is not None:
        translated_points_df = np.multiply(points_df, xyz1)

    translated_points_iou_df = None
    if points_iou_df is not None:
        translated_points_iou_df = np.multiply(points_iou_df, xyz1)

    if points_iou is not None:
        translated_points_iou = np.add(np.multiply(points_iou, xyz1), xyz2).astype(
            "float32"
        )
        return (
            translated_pointcloud,
            translated_points,
            translated_points_iou,
            translated_points_df,
            translated_points_iou_df,
        )
    else:
        return translated_pointcloud, translated_points, translated_points_df
