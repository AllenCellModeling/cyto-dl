import torch


class NoiseAnnealer:
    """Anneals variance of gaussian noise of real and fake examples passed to discriminator, called
    instance noise.

    This makes the generator's task harder and increases support of the real and fake distributions
    so they overlap, which has nice theoretical implications for the quality of the discriminator.
    Also can be used as a curriculum learning technique by iteratively unblurring the target to
    make segmentation harder over time.
    """

    def __init__(self, annealing_steps: int = 5000, init_variance: float = 0.3):
        """
        Parameters
        ----------
        annealing_steps:int=5000
            Number of steps to linearly anneal variance from `init_variance` to 0
        init_variance:float=0.3
            Initial variance of noise
        """
        self.init_variance = init_variance
        self.noise = init_variance
        self.step_size = init_variance / annealing_steps
        self.annealing_steps = annealing_steps
        self._done = False
        self.steps = 0

    def update_noise(self):
        if self.steps > self.annealing_steps:
            self._done = True
        else:
            self.noise -= self.step_size
            self.steps += 1

    def __call__(self, img):
        self.update_noise()
        if self._done:
            return img
        else:
            noise_tensor = torch.randn(img.shape) * self.noise
            noise_tensor = noise_tensor.type_as(img)
            return torch.add(img, noise_tensor)
