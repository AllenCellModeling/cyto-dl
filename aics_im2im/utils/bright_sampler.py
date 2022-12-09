import numpy as np


class BrightSampler:
    def __init__(self, key, threshold, base_prob, ch=0):
        self.key = key
        self.threshold = threshold
        self.base_prob = base_prob
        self.ch = ch

    def __call__(self, patch):
        return patch[self.key][self.ch].mean() > self.threshold or np.random.choice(
            [False, True], p=[1 - self.base_prob, self.base_prob]
        )
