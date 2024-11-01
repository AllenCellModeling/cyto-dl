from typing import Optional, Sequence, Union

import numpy as np
import torch
from monai.transforms import RandomizableTransform, Transform
from skimage.measure import regionprops


class CentroidCrop:
    """Class for cropping patches around passed centroids in an image.

    N
    """

    def __init__(self, crop_size: Sequence[str], remove_edge: bool = True):
        self.crop_size = crop_size
        self.remove_edge = remove_edge

    def centroid_to_slice(self, centroid):
        # slice across channel dimension
        slices = [slice(None, None)]
        for i, c in enumerate(centroid):
            start = int(c - self.crop_size[i] // 2)
            end = int(c + self.crop_size[i] // 2)
            slices.append(slice(start, end))
        return tuple(slices)

    def _filter_edge(self, centroids, shape, labels=None):
        assert len(shape) == len(
            self.crop_size
        ), "Image shape and crop_size must have the same dimensionality"
        valid_indices = [
            idx
            for idx, c in enumerate(centroids)
            # check distance to edges
            if all(
                [
                    c[i] >= self.crop_size[i] // 2 and c[i] <= (shape[i] - self.crop_size[i] // 2)
                    for i in range(len(c))
                ]
            )
        ]
        valid_centroids = np.array(centroids)[valid_indices]
        if labels is None:
            return valid_centroids, None
        return valid_centroids, np.array(labels)[valid_indices]

    def __call__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        centroids: Sequence[Sequence[int]],
        labels: Optional[Sequence[int]] = None,
        name="data",
    ):
        # don't include channel dimension in edge validation
        centroids, labels = self._filter_edge(centroids, data.shape[1:], labels)
        if len(centroids) == 0:
            raise ValueError("No valid centroids found")
        crops = [{name: data[self.centroid_to_slice(c)], "centroid": c} for c in centroids]
        if labels is not None:
            for crop, label in zip(crops, labels):
                crop[name] = label
        return crops


class CentroidCropd(CentroidCrop, Transform):
    """Transform for cropping patches around dictionary of images and corresponding centroids."""

    def __init__(
        self,
        keys: Sequence[str],
        crop_size: Sequence[int],
        centroid_key: str = "centroid",
        label_key: str = "label",
        remove_edge: bool = True,
    ):
        super().__init__(crop_size, remove_edge)
        self.keys = keys
        self.centroid_key = centroid_key
        self.label_key = label_key

    def __call__(self, data):
        centroids = data[self.centroid_key]
        labels = data[self.label_key] if self.label_key in data else None
        all_crops = None
        # data is C[Z]YX
        for k in self.keys:
            crops = super().__call__(data[k], centroids, labels, name=k)
            if not all_crops:
                all_crops = crops
            else:
                for i, crop in enumerate(crops):
                    all_crops[i][k] = crop[k]
        return all_crops


class SegCropd(RandomizableTransform):
    """Monai-style transform to crop a given size patch from an input image centered around each of
    the objects in an instance segmentation image."""

    def __init__(
        self,
        raw_keys: Union[str, Sequence[str]],
        seg_key: str,
        crop_size: Sequence[int],
        remove_edge: bool = True,
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.raw_keys = raw_keys if isinstance(raw_keys, (list, tuple)) else [raw_keys]
        self.seg_key = seg_key
        self.limit = limit

        self.cropper = CentroidCrop(crop_size, remove_edge)

    def get_centroids(self, seg):
        props = regionprops(seg)
        return [prop.centroid for prop in props], [prop.label for prop in props]

    def __call__(self, data):
        seg = data[self.seg_key].squeeze(0).astype(int)
        seg = seg.numpy() if isinstance(seg, torch.Tensor) else seg
        centroids, labels = self.get_centroids(seg)

        if self.limit is not None:
            idx = np.random.choice(len(centroids), self.limit)
            centroids = np.array(centroids)[idx]
            labels = np.array(labels)[idx]

        all_crops = None
        # data is C[Z]YX
        for k in self.raw_keys + [self.seg_key]:
            crops = self.cropper(data[k], centroids, labels, name=k)
            if not all_crops:
                all_crops = crops
            else:
                for i, crop in enumerate(crops):
                    all_crops[i][k] = crop[k]
        return all_crops
