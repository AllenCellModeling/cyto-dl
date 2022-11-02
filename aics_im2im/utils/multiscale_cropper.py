from monai.transforms import RandomizableTransform
import numpy as np


class RandomMultiScaleCropd(RandomizableTransform):
    def __init__(self, keys, patch_shape, patch_per_image, scales_dict):
        self.roi_size = np.asarray(patch_shape)
        self.keys = keys
        self.num_samples = patch_per_image
        self.scale_dict = scales_dict
        self.reversed_scale_dict = {}
        for k, v in scales_dict.items():
            for v_item in v:
                self.reversed_scale_dict[v_item] = k
        assert 1 in self.scale_dict.keys()

    def _generate_slice(self, start_coords, roi_size):
        return [slice(None, None)] + [
            slice(start, end)
            for start, end in zip(start_coords, start_coords + roi_size)
        ]

    def generate_multiscale_slices(self, start_indices):
        return {
            s: [
                self._generate_slice(
                    start_indices[s][i], (self.roi_size // s).astype(int)
                )
                for i in range(start_indices[s].shape[0])
            ]
            for s in start_indices.keys()
        }

    def generate_slices(self, image_dict):
        max_shape = np.asarray(image_dict[self.scale_dict[1][0]].shape[-3:])
        max_start_indices = max_shape - self.roi_size
        start_indices = np.asarray(
            [self.R.randint(max_start_indices) for _ in range(self.num_samples)]
        )
        scaled_start_indices = {
            s: (start_indices // s).astype(int) for s in self.scale_dict.keys()
        }
        return self.generate_multiscale_slices(scaled_start_indices)

    def __call__(self, image_dict):
        slices = self.generate_slices(image_dict)
        patches = []
        for i in self.num_samples:
            patch_dict = {
                key: data[slices[self.reversed_scale_dict[key][i]]]
                for key, data in image_dict.items()
                if key in self.keys
            }
            patches.append(patch_dict)
        return patches