from typing import List

from aicsimageio import AICSImage
from monai.data import MetaTensor
from monai.transforms import Transform


class AICSImageLoaderd(Transform):
    """Enumerates scenes and timepoints for dictionary with format.

    {path_key: path, channel_key: channel, scene_key: scene, timepoint_key: timepoint}.
    Differs from monai_bio_reader in that reading kwargs are passed in the dictionary, instead of
    fixed at initialization.
    """

    def __init__(
        self,
        path_key: str = "path",
        scene_key: str = "scene",
        kwargs_keys: List = ["dimension_order_out", "C", "T"],
        out_key: str = "raw",
        allow_missing_keys=False,
    ):
        """
        Parameters
        ----------
        path_key : str = "path"
            Key for the path to the image
        scene_key : str = "scene"
            Key for the scene number
        kwargs_keys : List = ["dimension_order_out", "C", "T"]
            Keys for the kwargs to pass to AICSImage.get_image_dask_data
        out_key : str = "raw"
            Key for the output image
        allow_missing_keys : bool = False
            Whether to allow missing keys in the data dictionary
        """
        super().__init__()
        self.path_key = path_key
        self.kwargs_keys = kwargs_keys
        self.allow_missing_keys = allow_missing_keys
        self.out_key = out_key
        self.scene_key = scene_key

    def __call__(self, data):
        # copying prevents the dataset from being modified inplace - important when using partially cached datasets so that the memory use doesn't increase over time
        data = data.copy()
        if self.path_key not in data and not self.allow_missing_keys:
            raise KeyError(f"Missing key {self.path_key} in data dictionary")
        path = data[self.path_key]
        img = AICSImage(path)
        if self.scene_key in data:
            img.set_scene(data[self.scene_key])
        kwargs = {k: data[k] for k in self.kwargs_keys}
        img = img.get_image_dask_data(**kwargs).compute()
        data[self.out_key] = MetaTensor(img, meta={"filename_or_obj": path, "kwargs": kwargs})

        return data
