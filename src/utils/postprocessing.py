import numpy as np
import torch
from skimage.exposure import rescale_intensity
from skimage.measure import label


def rescale(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu().numpy()
    return rescale_intensity(img, out_range=np.uint8).astype(np.uint8)


def detach(img: torch.Tensor) -> np.ndarray:
    return img.detach().cpu().numpy().astype(float)


def sigmoid_rescale(img: torch.Tensor) -> np.ndarray:
    img = torch.nn.Sigmoid()(img)
    img = img.detach().cpu().numpy()
    return (img * 255).astype(np.uint8)


def sigmoid_thresh(img: torch.Tensor) -> np.ndarray:
    img = torch.nn.Sigmoid()(img)
    img = img.detach().cpu().numpy()
    return (img > 123).astype(np.uint8)


def postprocess_label(img: torch.Tensor) -> np.ndarray:
    img = torch.nn.Sigmoid()(img)
    img = img.detach().cpu().numpy()
    return label(img > 0.5).astype(np.uint16)


class max_project:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, input_img: torch.Tensor):
        for key in self.keys:
            input_img[key], _ = torch.max(input_img[key].as_tensor(), 1)
        return input_img


def concat_dict(input_dict, keys):
    output_img = []
    for key in keys:
        im = input_dict[key].detach().cpu().numpy().astype(np.uint8)
        output_img.append(im)
    return np.stack(output_img)
