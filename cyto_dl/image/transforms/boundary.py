import torch
from monai.transforms import Transform
from omegaconf import ListConfig
from skimage.segmentation import find_boundaries


class FindBoundaries(Transform):
    """Transform for finding boundaries of a mask."""

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: Dict
            Additional arguments for skimage.segmentation.find_boundaries
        """
        super().__init__()
        self.kwargs = kwargs

    def __call__(self, img):
        return find_boundaries(img, **self.kwargs)


class FindBoundariesd(Transform):
    """Dictionary Transform for clipping image intensities based on absolute or percentile."""

    def __init__(
        self,
        keys: str,
        allow_missing_keys: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        keys: str
            name of images to find boundaries
        allow_missing_keys: bool
            whether to allow missing keys
        kwargs: Dict
            Additional arguments for skimage.segmentation.find_boundaries
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.allow_missing_keys = allow_missing_keys
        self.boundarizer = FindBoundaries(**kwargs)

    def __call__(self, img_dict):
        for key in self.keys:
            if key in img_dict.keys():
                img_dict[key] = self.boundarizer(img_dict[key])
            elif not self.allow_missing_keys:
                raise KeyError(f"key `{key}` not available. Available keys are {img_dict.keys()}")
        return img_dict
