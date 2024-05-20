from monai.transforms import Transform
import numpy as np
class IntensityCropZd(Transform):
    def __init__(self, keys, crop_size, channel=0):
        self.keys = keys
        self.crop_size = crop_size
        self.channel = channel
    
    def __call__(self, data):
        data = data.copy()
        for key in self.keys:
            img = data[key][self.channel]
            z_profile = img.mean(axis=(1, 2))
            # find crop_size window in z profile with max intensity
            window = np.ones(self.crop_size)
            convolved = np.convolve(z_profile, window, 'valid')
            start_index = np.argmax(convolved)
            slice_ = [slice(None, None), slice(start_index, start_index + self.crop_size), slice(None, None), slice(None, None)]

            data[key] = data[key][slice_]
        return data