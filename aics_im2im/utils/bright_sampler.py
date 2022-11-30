import numpy as np


class BrightSampler:
    def __init__(self, key, threshold, base_prob):
        self.key = key
        self.threshold = threshold
        self.base_prob = base_prob

    def __call__(self, patch):
        return patch[self.key].mean() > self.threshold or np.random.choice(
            [False, True], p=[1 - self.base_prob, self.base_prob]
        )
