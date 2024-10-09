import torch
from torch import nn


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor that has the same
    size as the input.
    """

    def __init__(
        self,
        gan_mode: str = "vanilla",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ):
        """Initialize the GANLoss class.

        Parameters
        ----------
            gan_mode:str='vanilla'
                Type of GAN objective `vanilla`, `lsgan`, and `wgangp` are supported.
            target_real_label:float=1.0
                label for a real image
            target_fake_label:float=0.0
                label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "wgangp":
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool):
        """Create label tensors with the same size as the input.

        Parameters
        ----------
            prediction:torch.Tensor
                Prediction output from a discriminator
            target_is_real:bool
                if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of input
        """
        target_tensor = self.real_label if target_is_real else self.fake_label
        target_tensor = target_tensor.expand_as(prediction)  # noqa: FURB184
        return target_tensor

    def __call__(self, prediction: torch.Tensor, target_is_real: bool):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters
        ----------
            prediction:torch.Tensor
                Prediction output from a discriminator
            target_is_real:bool
                if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ("lsgan", "vanilla"):
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


# modified from https://github.com/MMV-Lab/mmv_im2im/blob/1b92bf4ab27cafe2608aef071f366741df3b58d4/mmv_im2im/utils/gan_losses.py
class Pix2PixHD(nn.Module):
    def __init__(self, scales, loss_weights={"GAN": 1, "FM": 10}):
        super().__init__()
        self.scales = scales
        self.gan_loss = GANLoss("vanilla")
        self.feature_matching_loss = torch.nn.L1Loss()
        self.weights = loss_weights

    def get_feature_matching_loss(self, features):
        loss_fm = 0
        for scale in range(self.scales):
            for real_feat, pred_feat in zip(features["real"][scale], features["pred"][scale]):
                loss_fm += self.feature_matching_loss(real_feat.detach(), pred_feat)
        return loss_fm / self.scales

    def get_gan_loss(self, features, feature_type):
        loss = 0
        for scale in range(self.scales):
            loss += self.gan_loss(features[scale][-1], feature_type == "real")
        return loss / self.scales

    def __call__(self, features, step):
        if step == "discriminator":
            return (
                self.get_gan_loss(features["real"], "real")
                + self.get_gan_loss(features["pred"], "pred")
            ) * 0.5

        elif step == "generator":
            # tell discriminator these are real features
            loss_G = self.get_gan_loss(features["pred"], "real")
            loss_feature_matching = self.get_feature_matching_loss(features)
            return loss_G * self.weights["GAN"] + loss_feature_matching * self.weights["FM"]
