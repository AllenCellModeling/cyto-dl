import numpy as np
import torch
from skimage.exposure import rescale_intensity


def rescale(img):
    img = img.detach().cpu().numpy()
    return rescale_intensity(img, out_range=np.uint8).astype(np.uint8)


def sigmoid_rescale(img):
    img = torch.nn.Sigmoid()(img)
    img = img.detach().cpu().numpy()
    return (img * 255).astype(np.uint8)


def sigmoid_thresh(img):
    img = torch.nn.Sigmoid()(img)
    img = img.detach().cpu().numpy()
    return (img > 123).astype(np.uint8)


def postprocess_label(img):
    from skimage.measure import label

    img = torch.nn.Sigmoid()(img)
    img = img.detach().cpu().numpy()
    return label(img > 0.5).astype(np.uint16)


class max_project:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, input_img):
        for key in self.keys:
            input_img[key], _ = torch.max(input_img[key].as_tensor(), 1)
        return input_img
