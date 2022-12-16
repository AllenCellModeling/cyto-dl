import torch

class NoiseAnnealer:
    """
    Anneals variance of gaussian noise of real and fake examples passed to discriminator, called instance noise.
    This makes the generator's task harder and increases support of the real and fake distributions so they overlap, which
    has nice theoretical implications for the quality of the discriminator.
    Also can be used as a curriculum learning technique by iteratively unblurring the target to make segmentation harder over time.
    """

    def __init__(self, annealing_steps=5000, init_variance=0.3):
        self.init_variance = init_variance
        self.noise = 0
        self.annealing_steps = annealing_steps

    def update_noise(self, step):
        if step > self.annealing_steps:
            self.noise = 0
        else:
            self.noise = self.init_variance * (1 - (step / self.annealing_steps))

    def add_noise(self, img):
        if self.noise > 0:
            noise_tensor = torch.randn(img.shape) * self.noise
            noise_tensor = noise_tensor.type_as(img)
            return torch.add(img, noise_tensor)
        else:
            return img
