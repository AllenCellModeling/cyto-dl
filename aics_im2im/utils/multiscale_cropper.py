import numpy as np


import numpy as np

class MultiScaleCropper:
    def __init__(self, image_shape, roi_size: tuple, scales: list, num_samples: int):
        roi_size = np.asarray(roi_size)
        self.dims = len(image_shape)
        self.max_start = np.asarray(image_shape) - roi_size
        self.num_samples = num_samples
        self.scales = scales

        start_indices = self.generate_start_indices()
        scaled_start_indices = self.generate_scaled_start_indices(start_indices)
        self.slices = self.generate_multiscale_slices(scaled_start_indices)

    def generate_scaled_start_indices(self, start_indices):
        return {
            s: (start_indices//s).astype(int) for s in self.scales
        }

    def generate_start_indices(self):
        return np.asarray([np.random.randint(self.max_start) for _ in range(self.num_samples)])

    def generate_slice(self, start_coords, roi_size):
        return [slice(None, None)] + [slice(start, end) for start, end in zip(start_coords, start_coords + roi_size)]

    def generate_multiscale_slices(self, start_indices, roi_size):
        return {
            s: [self.generate_slice(start_indices[s][i], (roi_size//s).astype(int)) for i in range(start_indices.shape[0])] for  s in self.scales
        }

    def __call__(self, image, metadata):
        patches ={}
        for key, data in image.items():
            patches[key] = [data[scale_slice] for scale_slice in self.slices[metadata[key]['patch_ratio']]]
        return patches

