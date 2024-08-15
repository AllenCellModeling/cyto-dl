from typing import Sequence

from monai.data import MetaTensor
from monai.transforms import Transform


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
