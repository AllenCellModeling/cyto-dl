from typing import Union

import numpy as np
from monai.transforms import Transform
from skimage.draw import polygon2mask


class PolygonLoaderd(Transform):
    """Convert numpy array of polygon vertices to a torch tensor."""

    def __init__(
        self,
        keys: Union[list, str],
        shape_reference_key: str,
        propagate_3d: bool = True,
        missing_key_mode: str = "raise",
    ):
        """
        Parameters
        ----------
        keys: Union[list, str]
            keys containing polygon paths
        shape_reference_key: str
            key to base mask shape on
        propagate_3d: bool=True
            Whether to propagate 2D mask to 3D. Currently, only True is supported.
        missing_key_mode: str='raise'
            How to handle missing keys. Options are 'raise', 'ignore', and 'create'. Raise will raise a KeyError, ignore will do nothing, and create will create a new key with a blank mask.
        """
        super().__init__()
        self.keys = keys
        self.shape_reference_key = shape_reference_key
        assert missing_key_mode in ("ignore", "raise", "create")
        self.missing_key_mode = missing_key_mode
        self.propagate_3d = propagate_3d
        if not self.propagate_3d:
            raise NotImplementedError("propagate_3d=False not implemented")

    def __call__(self, input_dict):
        """
        Parameters
        ----------
        input_dict: Dict[str, torch.Tensor]
            dict of CZYX tensors/metadata/paths
        """
        for key in self.keys:
            if key in input_dict.keys() and input_dict[key] is not None:
                poly = np.load(input_dict[key], allow_pickle=True)

                mask_shape = input_dict[self.shape_reference_key].shape[-2:]
                mask = np.zeros(mask_shape)
                for p in poly:
                    mask = np.logical_or(mask, polygon2mask(mask_shape, p))
                if self.propagate_3d:
                    mask = np.stack([mask] * input_dict[self.shape_reference_key].shape[1])
                # all ones except for regions in polygon
                mask = ~mask
                input_dict[key] = np.expand_dims(mask > 0, 0)
            elif self.missing_key_mode == "raise":
                raise KeyError(
                    f"key `{key}` not available. Available keys are {input_dict.keys()}"
                )
            elif self.missing_key_mode == "create":
                input_dict[key] = np.ones_like(input_dict[self.shape_reference_key])
        return input_dict
