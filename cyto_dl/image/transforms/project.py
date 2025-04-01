from typing import Union

from monai.transforms import Transform
from monai.transforms.utils_pytorch_numpy_unification import max as _max
from monai.transforms.utils_pytorch_numpy_unification import mean, median
from monai.transforms.utils_pytorch_numpy_unification import min as _min
from monai.transforms.utils_pytorch_numpy_unification import mode, std
from omegaconf import ListConfig


class Projectd(Transform):  # codespell:ignore
    """Monai-style transform to apply projections (e.g., max, std) to an image."""

    def __init__(
        self,
        keys: Union[list, str],
        projection_dim: int = 1,
        projection_type: str = "max",
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys: Union[list, str]
            keys to apply projection
        projection_dim: int=1
            index into C[Z]YX to compute projection across
        projection_type: str="max"
            Type of projection to apply. Options: "max", "min", "std", "median", "mode", "mean"
        allow_missing_keys: bool=False
            Whether to raise error if specified key is missing
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.projection_dim = projection_dim
        self.allow_missing_keys = allow_missing_keys

        projection_fns = {
            "max": _max,
            "min": _min,
            "std": std,
            "median": median,
            "mode": mode,
            "mean": mean,
        }

        if projection_type not in projection_fns:
            raise ValueError(
                f"Unsupported projection_type: {projection_type}. Supported types: {projection_fns.keys()}"
            )
        self.projector = projection_fns[projection_type]

    def __call__(self, input_dict):
        """
        Parameters
        ----------
        input_dict: Dict[str, torch.Tensor]
            dict of C[Z]YX tensors
        """
        for key in self.keys:
            if key in input_dict.keys():
                input_dict[key] = self.projector(input_dict[key], dim=self.projection_dim)
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"key `{key}` not available. Available keys are {input_dict.keys()}"
                )
        return input_dict
