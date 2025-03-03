import re
from typing import List

import numpy as np
from bioio import BioImage
from monai.data import MetaTensor
from monai.transforms import Transform

from cyto_dl.utils.arg_checking import get_dtype


class BioIOImageLoaderd(Transform):
    """Enumerates scenes and timepoints for dictionary with format.

    {path_key: path, channel_key: channel, scene_key: scene, timepoint_key: timepoint}.
    Differs from monai_bio_reader in that reading kwargs are passed in the dictionary, instead of fixed at
    initialization. The filepath will be saved in the dictionary as 'filename_or_obj' (with or without metadata depending on `include_meta_in_filename`).
    """

    def __init__(
        self,
        path_key: str = "path",
        scene_key: str = "scene",
        resolution_key: str = "resolution",
        kwargs_keys: List[str] = ["dimension_order_out", "C", "T"],
        out_key: str = "raw",
        allow_missing_keys=False,
        dtype: np.dtype = np.float16,
        dask_load: bool = True,
        include_meta_in_filename: bool = False,
    ):
        """
        Parameters
        ----------
        path_key : str = "path"
            Key for the path to the image
        scene_key : str = "scene"
            Key for the scene number
        kwargs_keys : List = ["dimension_order_out", "C", "T"]
            Keys for the kwargs to pass to BioImage.get_image_dask_data. Values in the csv can be comma separated list.
        out_key : str = "raw"
            Key for the output image
        allow_missing_keys : bool = False
            Whether to allow missing keys in the data dictionary
        dtype : np.dtype = np.float16
            Data type to cast the image to
        dask_load: bool = True
            Whether to use dask to load images. If False, full images are loaded into memory before extracting specified scenes/timepoints.
        include_meta_in_filename: bool = False
            Whether to include metadata in the filename. Useful when loading multi-dimensional images with different kwargs.
        """
        super().__init__()
        self.path_key = path_key
        self.kwargs_keys = kwargs_keys
        self.allow_missing_keys = allow_missing_keys
        self.out_key = out_key
        self.resolution_key = resolution_key
        self.scene_key = scene_key
        self.dtype = get_dtype(dtype)
        self.dask_load = dask_load
        self.include_meta_in_filename = include_meta_in_filename

    def split_args(self, arg):
        if isinstance(arg, str) and "," in arg:
            return list(map(int, arg.split(",")))
        return arg

    def _get_filename(self, path, kwargs):
        if self.include_meta_in_filename:
            path = path.split(".")[0] + "_" + "_".join([f"{k}_{v}" for k, v in kwargs.items()])
        # remove illegal characters from filename
        path = re.sub(r'[<>:"|?*]', "", path)
        return path

    def __call__(self, data):
        # copying prevents the dataset from being modified inplace - important when using partially cached datasets so that the memory use doesn't increase over time
        data = data.copy()
        if self.path_key not in data and not self.allow_missing_keys:
            raise KeyError(f"Missing key {self.path_key} in data dictionary")
        path = data[self.path_key]
        img = BioImage(path)
        if self.scene_key in data:
            img.set_scene(data[self.scene_key])
        if self.resolution_key in data:
            img.set_resolution_level(data[self.resolution_key])
        kwargs = {k: self.split_args(data[k]) for k in self.kwargs_keys if k in data}
        if self.dask_load:
            img = img.get_image_dask_data(**kwargs).compute()
        else:
            img = img.get_image_data(**kwargs)
        img = img.astype(self.dtype)
        if self.scene_key in data:
            kwargs["scene"] = data[self.scene_key]
        kwargs.update({"filename_or_obj": self._get_filename(path, kwargs)})

        data[self.out_key] = MetaTensor(img, meta=kwargs)
        return data
