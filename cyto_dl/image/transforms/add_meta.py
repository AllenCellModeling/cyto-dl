from typing import Sequence

from monai.data import MetaTensor
from monai.transforms import Transform
from omegaconf import ListConfig


class AddMeta(Transform):
    """Transform to update image key metadata with new keys."""

    def __init__(self, image_key: str, meta_keys: Sequence[str], delete: bool = False):
        """
        Parameters
        ----------
        image_key: str
            Key in batch dictionary for image data. Must be a MetaTensor
        meta_keys: Sequence[str]
            List of keys to add to image metadata
        delete: bool
            If True, delete the original meta_keys from the image metadata after they have been added as metadata
        """
        self.image_key = image_key
        self.meta_keys = meta_keys
        self.delete = delete

    def __call__(self, data):
        if not isinstance(data[self.image_key], MetaTensor):
            raise ValueError(
                f"Image key {self.image_key} must be a MetaTensor, got {type(data[self.image_key])}"
            )
        new_meta = {k: data[k] for k in self.meta_keys}
        if self.delete:
            for k in self.meta_keys:
                del data[k]
        data[self.image_key].meta.update(new_meta)
        return data


class MetaToKey(Transform):
    """Transform to add metadata from image key to the batch dictionary."""

    def __init__(self, image_key: str, meta_keys: Sequence[str], replace: bool = False):
        """
        Parameters
        ----------
        image_key: str
            Key in batch dictionary for image data. Must be a MetaTensor
        meta_keys: Sequence[str]
            List of keys to add to batch dictionary
        replace: bool
            If True, replace meta_keys in batch dictionary with those from image metadata if they already exist
        """
        self.meta_keys = meta_keys if isinstance(meta_keys, (list, ListConfig)) else [meta_keys]
        self.image_key = image_key
        self.replace = replace

    def __call__(self, data):
        for k in self.meta_keys:
            if not isinstance(data[self.image_key], MetaTensor):
                raise ValueError(
                    f"Image key {self.image_key} must be a MetaTensor, got {type(data[self.image_key])}"
                )
            if k in data and not self.replace:
                raise ValueError(f"Key {k} already exists in batch dictionary")
            if k not in data[self.image_key].meta:
                raise ValueError(f"Key {k} not found in image metadata")
            data[k] = data[self.image_key].meta[k]
        return data
