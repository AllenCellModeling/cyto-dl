import numpy as np


class DarkSampler:
    """Class to decide whether to sample a given patch based on its mean intensity + a background
    sampling rate."""

    def __init__(self, key: str, threshold: float, base_prob: float, ch: int = 0):
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
        self.key = key
        self.threshold = threshold
        self.base_prob = base_prob
        self.ch = ch

    def __call__(self, patch):
        return patch[self.key][self.ch].half().mean() <= self.threshold or np.random.choice(
            [False, True], p=[1 - self.base_prob, self.base_prob]
        )
