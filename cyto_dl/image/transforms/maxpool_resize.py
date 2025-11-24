from typing import Sequence, Union
from monai.transforms import Transform
from monai.utils import ensure_tuple_rep
from omegaconf import ListConfig
import torch
import torch.nn.functional as F

class MaxPoolResized(Transform):
    """Resizes images or volumes using max pooling over spatial dimensions.

    This transform applies adaptive max pooling to reduce the spatial dimensions of the input
    tensor to the specified `spatial_size`. It supports:
    - 3D tensors ([channels, height, width]) for single images.
    - 4D tensors ([batch, channels, height, width]) for batched images.
    - 5D tensors ([batch, channels, depth, height, width]) for volumetric data.

    The transform uses `torch.nn.functional.adaptive_max_pool2d` for 2D data and
    `adaptive_max_pool3d` for 3D data to ensure precise output sizes. Non-positive values
    in `spatial_size` are replaced with the corresponding input dimensions.

    Parameters
    ----------
    keys: Union[str, Sequence[str]]
        Keys of the corresponding items to be transformed in the input dictionary.
    spatial_size: Union[Sequence[int], int]
        Expected shape of spatial dimensions after resize operation.
        If a single integer is provided, it is applied to all spatial dimensions
        (e.g., 32 -> (32, 32) for 2D or (32, 32, 32) for 3D).
        If a sequence is provided, its length must match the number of spatial dimensions
        (2 for images, 3 for volumes).
    allow_missing_keys: bool, optional
        If True, skips missing keys in the input data without raising an error.
        Default is False.

    Raises
    ------
    TypeError
        If the input for a key is not a PyTorch tensor.
    ValueError
        If the input tensor has an unsupported number of dimensions, if `spatial_size`
        has an invalid length, or if the output spatial size is invalid (e.g., larger
        than the input size).
    KeyError
        If a key is missing in the input data and `allow_missing_keys` is False.
 
    """

    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        spatial_size: Union[Sequence[int], int],
        allow_missing_keys: bool = False,
    ):
        super().__init__()
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.allow_missing_keys = allow_missing_keys

        # Store raw spatial_size for later validation
        self.spatial_size = spatial_size

        # Validate spatial_size length in advance
        if isinstance(spatial_size, (list, tuple, ListConfig)):
            if len(spatial_size) not in (2, 3):
                raise ValueError(
                    f"spatial_size sequence must have length 2 or 3, got length {len(spatial_size)}"
                )

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key '{key}' not found in input data.")

            x = d[key]
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Input '{key}' must be a PyTorch tensor, got {type(x)}")

            # Determine expected tensor dimensions and spatial size length
            input_dims = x.dim()
            if input_dims == 3:  # [C, H, W]
                expected_spatial_dims = 2
                x = x.unsqueeze(0)  # Add batch dimension: [1, C, H, W]
                squeeze = True
            elif input_dims == 4:  # [B, C, H, W]
                expected_spatial_dims = 2
                squeeze = False
            elif input_dims == 5:  # [B, C, D, H, W]
                expected_spatial_dims = 3
                squeeze = False
            else:
                raise ValueError(
                    f"Input '{key}' must be a 3D ([C, H, W]), 4D ([B, C, H, W]), "
                    f"or 5D ([B, C, D, H, W]) tensor, got shape {x.shape}"
                )

            # Normalize spatial_size to match expected number of spatial dimensions
            try:
                spatial_size = ensure_tuple_rep(self.spatial_size, expected_spatial_dims)
            except ValueError as e:
                raise ValueError(
                    f"spatial_size sequence must have length {expected_spatial_dims}, got {self.spatial_size}"
                ) from e

            orig_size = x.shape[-expected_spatial_dims:]

            # Replace non-positive spatial_size values with original dimensions
            spatial_size = [
                orig if s <= 0 else s
                for s, orig in zip(spatial_size, orig_size)
            ]

            # Validate output spatial size
            for i, (s, orig) in enumerate(zip(spatial_size, orig_size)):
                if s > orig:
                    raise ValueError(
                        f"Output spatial size {spatial_size} for dimension {i} "
                        f"exceeds input size {orig_size} for key '{key}'"
                    )
                if s <= 0:
                    raise ValueError(
                        f"Output spatial size {s} for dimension {i} must be positive "
                        f"for key '{key}'"
                    )

            # Apply max pooling based on input dimensions
            if expected_spatial_dims == 2:
                x = F.adaptive_max_pool2d(x, output_size=spatial_size)
            else:  # expected_spatial_dims == 3
                x = F.adaptive_max_pool3d(x, output_size=spatial_size)

            # Remove batch dimension for 3D inputs
            if squeeze:
                x = x.squeeze(0)

            d[key] = x

        return d
