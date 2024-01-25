import copy
from typing import List, Union

import numpy as np
import torch
from monai.transforms import RandomizableTransform, Resize, Transform
from omegaconf import ListConfig


class GenerateTrackLabels(Transform):
    """Transform to generate track labels from breakdown and formation labels."""

    def __init__(
        self,
        img_key: str = "img",
        formation_key: str = "formation",
        breakdown_key: str = "breakdown",
        track_start_key: str = "track_start",
        label_key: str = "label",
    ):
        """
        Parameters
        ----------
        img_key: str
            Key for column with image
        formation_key: str
            Key for column with formation
        breakdown_key: str
            Key for column with breakdown
        track_start_key: str
            Key for column with track start
        label_key: str
            Key to save label into
        """
        super().__init__()
        self.formation_key = formation_key
        self.breakdown_key = breakdown_key
        self.track_start_key = track_start_key
        self.label_key = label_key

    def __call__(self, img_dict):
        n_timepoints = img_dict[self.img_key].shape[0]
        formation_idx = int(img_dict[self.formation_key] - img_dict[self.track_start])
        breakdown_idx = int(img_dict[self.breakdown_key] - img_dict[self.track_start])

        # 0: normal, 1: mitotic
        tp_labels = np.zeros(n_timepoints)
        if 0 <= formation_idx < len(tp_labels):
            tp_labels[:formation_idx] = 1

        if 0 <= breakdown_idx < len(tp_labels):
            tp_labels[breakdown_idx + 1 :] = 1
        img_dict[self.label_key] = tp_labels
        return img_dict


class TrackCrop(RandomizableTransform):
    """Transform to randomly crop track to first n or last n timepoints."""

    def __init__(
        self,
        img_key: str = "img",
        label_key: str = "label",
        p_first: float = 0.15,
        p_last: float = 0.1,
        percentage=[0.2, 0.7],
        max_crop_length: int = 20,
    ):
        """
        Parameters
        ----------
        img_key: str
            Key for column with image
        p_first: float
            Probability of cropping to start of track
        p_last: float
            Probability of cropping to end of track
        percentage: list
            Range of percentages of track length to crop
        max_crop_length: int
            Maximum number of timepoints to crop
        """
        super().__init__()
        self.img_key = img_key
        self.label_key = label_key
        self.p_first = p_first
        self.p_last = p_last
        self.percentage = np.array(percentage)
        self.max_crop_length = max_crop_length

    def __call__(self, img_dict):
        new_im_dict = copy.deepcopy(img_dict)
        new_im = img_dict[self.img_key]

        n = self.R.randint(*(self.percentage * new_im.shape[0]).astype(int), size=1)[0]
        n = min(self.max_crop_length, n)
        # crop to start of track
        if self.R.random() < self.p_first:
            new_im_dict[self.img_key] = new_im[:n]
            new_im_dict[self.label_key] = new_im_dict[self.label_key][:n]
        # crop to end of track
        elif self.R.random() < self.p_last:
            new_im_dict[self.img_key] = new_im[-n:]
            new_im_dict[self.label_key] = new_im_dict[self.label_key][-n:]
        return new_im_dict


class PerChannel(Transform):
    """Transform to apply same transform to each channel of image."""

    def __init__(
        self,
        keys: Union[str, List, ListConfig],
        transform: Transform,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys: list
            List of keys to apply transform to
        transform: Transform
            Transform to apply to each channel
        allow_missing_keys: bool
            Whether to allow missing keys
        """
        super().__init__()
        self.transform = transform
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, img_dict):
        new_im_dict = copy.deepcopy(img_dict)
        for key in self.keys:
            if key not in new_im_dict and not self.allow_missing_keys:
                raise KeyError(
                    f"Key {key} not found in image dictionary. Available keys are {list(new_im_dict.keys())}"
                )
            for i in range(new_im_dict[key].shape[0]):
                new_im_dict[key][i] = self.transform(new_im_dict[key][i])
        return new_im_dict


class CropResize(RandomizableTransform):
    def __init__(
        self, keys: Union[str, List, ListConfig], max_shift=8, allow_missing_keys: bool = False
    ):
        """
        Parameters
        ----------
        keys: list
            List of keys to apply transform to
        max_shift: int
            Maximum number of pixels to shift image by before resizing
        allow_missing_keys: bool
            Whether to allow missing keys
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.max_shift = max_shift
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, img_dict):
        breakpoint()
        new_im_dict = copy.deepcopy(img_dict)

        for key in self.keys:
            if key not in new_im_dict and not self.allow_missing_keys:
                raise KeyError(
                    f"Key {key} not found in image dictionary. Available keys are {list(new_im_dict.keys())}"
                )
            resizer = Resize(new_im_dict[key].shape[-2:])
            resized_movie = []
            for im in new_im_dict[key]:
                shift = self.R.randint(0, self.max_shift, size=4)
                im = im[shift[0] : im.shape[0] - shift[1], shift[2] : im.shape[1] - shift[3]]
                im = resizer(im.unsqueeze(0)).squeeze(0)
                resized_movie.append(im)
            new_im_dict[key] = torch.stack(resized_movie)
        return new_im_dict
