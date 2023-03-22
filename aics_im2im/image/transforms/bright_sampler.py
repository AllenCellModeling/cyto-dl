import numpy as np
from omegaconf import ListConfig


class BrightSampler:
    """Class to decide whether to sample a given patch based on its mean intensity + a background
    sampling rate."""

    def __init__(self, keys: str, threshold: float, base_prob: float, ch: int = 0):
        """
        Parameters
        ----------
        key: str
            name of image to be examined
        threshold: float
            threshold for selecting patch
        base_prob:
            background probabiliy of selecting a patch
        ch:
            channel from image to check against threshold
        """
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.threshold = threshold
        self.base_prob = base_prob
        self.ch = ch

    def __call__(self, patch):
        for key in self.keys:
            if key in patch.keys():
                break
        else:
            raise ValueError(
                f"None of keys {self.keys} found in data. Available keys are {patch.keys()}"
            )
        return patch[key][self.ch].float().mean() > self.threshold or np.random.choice(
            [False, True], p=[1 - self.base_prob, self.base_prob]
        )
