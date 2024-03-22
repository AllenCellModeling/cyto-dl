import numpy as np
from monai.transforms import Transform
from skimage.exposure import match_histograms
from skimage.filters import gaussian
from skimage.transform import resize


class Crappifyd(Transform):
    """Crappify 100x images to look like 20x images.

    This eliminates the need for perfect alignment in transfer function training
    """

    def __init__(
        self,
        hr_key: str,
        lr_key: str,
    ):
        """
        Parameters
        ----------
        keys: str
            name of images to resize

        low: float
            lower bound for clipping
        high: float
            upper bound for clipping
        percentile: bool
            whether to use percentile or absolute values  for clipping
        allow_missing_keys: bool
            whether to fail if provided keys are missing
        """
        super().__init__()
        self.hr_key = hr_key
        self.lr_key = lr_key

    def add_salt_and_pepper_noise(self, image, salt_probability=0.01, pepper_probability=0.01):
        # Salt noise addition
        num_salt = np.ceil(salt_probability * image.size)
        salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[1:]]
        image[:, salt_coords[0], salt_coords[1], salt_coords[2]] = np.percentile(image, 99)

        # Pepper noise addition
        num_pepper = np.ceil(pepper_probability * image.size)
        pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[1:]]
        image[:, pepper_coords[0], pepper_coords[1], pepper_coords[2]] = np.percentile(image, 1)

        return image

    def __call__(self, img_dict):
        hr_img = img_dict[self.hr_key].numpy()
        lr_img = img_dict[self.lr_key].numpy()
        # rescale to 20x
        crap_img = resize(hr_img, lr_img.shape, anti_aliasing=True, preserve_range=True)
        # blur, esp. in z
        crap_img = gaussian(crap_img, sigma=[1, 2, 1.5, 1.5], preserve_range=True)
        # add gaussian noise
        crap_img += np.random.normal(
            np.median(crap_img), crap_img.std(), size=crap_img.shape
        ).clip(0, hr_img.max())
        # add poisson noise
        crap_img = np.random.poisson(crap_img).clip(0, hr_img.max())
        # add salt and pepper noise
        crap_img = self.add_salt_and_pepper_noise(crap_img)
        # adjust contrast
        crap_img = crap_img**0.5
        # match histogram
        crap_img = match_histograms(crap_img, lr_img)

        img_dict[self.lr_key] = crap_img

        return img_dict
