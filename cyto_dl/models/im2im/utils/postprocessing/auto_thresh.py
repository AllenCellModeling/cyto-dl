import importlib
from typing import Optional, Union

import numpy as np


class AutoThreshold:
    def __init__(self, method: Optional[Union[float, str]] = None):
        if isinstance(method, float):

            def thresh_func(image):
                return method

        elif isinstance(method, str):
            try:
                thresh_func = getattr(importlib.import_module("skimage.filters"), method)
            except AttributeError:
                raise AttributeError(f"method {method} not found in skimage.filters")
        elif method is None:
            thresh_func = None
        else:
            raise TypeError("method must be a float or a string")
        self.thresh_func = thresh_func

    def __call__(self, image):
        image = image.detach().cpu().float().numpy()
        if self.thresh_func is None:
            return image
        return (image > self.thresh_func(image)).astype(np.uint8)
