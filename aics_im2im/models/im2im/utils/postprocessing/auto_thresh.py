import importlib
from typing import Union


class AutoThreshold:
    def __init__(self, method: Union[float, str]):
        if isinstance(method, float):

            def thresh_func(image):
                return method

        elif isinstance(method, str):
            try:
                thresh_func = getattr(importlib.import_module("skimage.filters"), method)
            except AttributeError:
                raise AttributeError(f"method {method} not found in skimage.filters")
        else:
            raise TypeError("method must be a float or a string")
        self.thresh_func = thresh_func

    def __call__(self, image):
        return image > self.thresh_func(image)
