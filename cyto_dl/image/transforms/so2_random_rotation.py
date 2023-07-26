import torch
from monai.transforms import Transform

from cyto_dl.models.vae.so2_image_vae.utils import get_rotation_matrix, rotate_img


class SO2RandomRotate(Transform):
    def __init__(
        self,
        spatial_dims: int,
    ):
        """Random rotate input on the XY plane. Assumes ZYX or ZXY ordering of coordinates if 3d.

        Parameters
        ----------
        spatial_dims: int
            Whether 2d or 3d
        """
        super().__init__()
        self.spatial_dims = spatial_dims

    def __call__(self, img):
        angle = torch.rand(1) * torch.pi * 2
        angle = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
        rot = get_rotation_matrix(angle, spatial_dims=self.spatial_dims)

        to_squeeze = False
        if len(img.shape) == (self.spatial_dims + 1):
            img = img.unsqueeze(0)
            to_squeeze = True

        img = rotate_img(img, rot.type_as(img))
        if to_squeeze:
            return img.squeeze(0)
        return img


class SO2RandomRotated(Transform):
    def __init__(
        self,
        keys,
        spatial_dims: int,
    ):
        """Dictionary-transform version of SO2RandomRotate."""
        super().__init__()
        self.keys = keys
        self.transform = SO2RandomRotate(spatial_dims)

    def __call__(self, img):
        for key in self.keys:
            img[key] = self.transform(img[key])

        return img
