from typing import Union

from monai.transforms import Transform


class Merged(Transform):
    """Use mask to merge two images."""

    def __init__(
        self,
        mask_key: str,
        image_keys: Union[list, str],
        base_image_key: str,
        output_name: str,
    ):
        """
        Parameters
        ----------
        mask_key: str
            key for mask
        image_keys: Union[list, str]
            keys to merge
        base_image_key: str
            name of image to serve as base image
        """
        super().__init__()
        self.mask_key = mask_key
        if len(image_keys) != 2:
            raise ValueError(f"image_keys must be a list of length 2. Got {image_keys}")
        self.image_keys = list(image_keys)
        self.base_image_key = base_image_key
        self.output_name = output_name

    def __call__(self, input_dict):
        """
        Parameters
        ----------
        input_dict: Dict[str, torch.Tensor]
            dict of CZYX tensors/metadata/paths
        """
        # no merging mask, return original dict
        if self.mask_key not in input_dict or input_dict[self.mask_key] is None:
            return input_dict

        mask = input_dict[self.mask_key]

        if self.base_image_key not in input_dict:
            raise KeyError(
                f"key `{self.base_image_key}` not available. Available keys are {input_dict.keys()}"
            )
        base_image_name = input_dict[self.base_image_key]

        if base_image_name not in self.image_keys:
            raise KeyError(
                f"Base image name `{base_image_name}` must match provided image keys `{self.image_keys}`"
            )

        for key in self.image_keys:
            if key not in input_dict.keys():
                raise KeyError(
                    f"key `{key}` not available. Available keys are {input_dict.keys()}"
                )

        base_image = input_dict[base_image_name]
        self.image_keys.remove(base_image_name)
        merge_image = input_dict[self.image_keys[0]]

        input_dict[self.output_name] = (base_image * ~mask) + (merge_image * mask)

        return input_dict
