import pytest
import torch
from monai.transforms import Flip

from cyto_dl.models.im2im.utils.omnipose import OmniposeRandFlipd


@pytest.mark.parametrize("spatial_axis", [0, 1, 2])
def test_gradient_flip(spatial_axis):
    # flip image and compute gradient
    img = torch.rand((30, 30, 30))
    flipper = Flip(spatial_axis=spatial_axis)
    # transforms expects CZYX tensor
    img_flip = flipper(img.unsqueeze(0))
    img_flip_grad = torch.stack(torch.gradient(img_flip.squeeze(0)))

    # compute gradient and flip image
    grad = torch.stack(torch.gradient(img))
    # put grad image into fake omnipose-generated gt
    omnipose_im = torch.ones((7, 30, 30, 30))
    omnipose_im[3:6] = grad
    grad_flipper = OmniposeRandFlipd(label_keys=["im"], spatial_axis=spatial_axis, prob=1.0)
    flip_im = grad_flipper({"im": omnipose_im})["im"]
    # extract gradient
    flip_grad = flip_im[3:6]

    assert torch.equal(img_flip_grad, flip_grad)
