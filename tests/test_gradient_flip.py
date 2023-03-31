import pytest
import torch
from monai.transforms import Flip

from aics_im2im.image.transforms.rand_flip_gradient import RandFlipGrad


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
    grad_flipper = RandFlipGrad(spatial_axis=spatial_axis, prob=1.0)
    flip_grad = grad_flipper(grad)

    assert torch.equal(img_flip_grad, flip_grad)
