import torch
import torch.nn as nn
from einops import rearrange, repeat

from cyto_dl.models.jepa import JEPABase


class IWM(JEPABase):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        predictor: nn.Module,
        source_key: str = "source",
        target_key: str = "target",
        target_domain_key: str = "target_domain",
        save_dir: str = "./",
        momentum: float = 0.998,
        max_epochs: int = 100,
        predict_target: bool = False,
        **base_kwargs,
    ):
        """Image World Model for self-supervised learning of encoder and predictor of image
        translation in latent space.

        Parameters
        ----------
        encoder : nn.Module
            The encoder module used for feature extraction.
        predictor : nn.Module
            The predictor module used for generating predictions.
        source_key : str
            The key used to access the source data.
        target_key : str
            The key used to access the target data.
        target_domain_key : str
            The key used to access the target domain data.
        save_dir : str, optional
            The directory to save the model predictions (default is "./").
        momentum : float, optional
            The momentum value for the exponential moving average of the model weights (default is 0.998).
        max_epochs : int, optional
            The maximum number of training epochs (default is 100).
        predict_target : bool, optional
            Whether to predict the target embeddings instead of just extracting embeddings of source image (default is False).
        **base_kwargs : dict
            Additional arguments passed to the BaseModel.
        """
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            x_key=source_key,
            save_dir=save_dir,
            momentum=momentum,
            max_epochs=max_epochs,
            **base_kwargs,
        )

    def model_step(self, stage, batch, batch_idx):
        self.update_teacher()
        source = batch[self.hparams.source_key]
        target = batch[self.hparams.target_key]

        target_masks = self.get_mask(batch, "target_mask")
        context_masks = self.get_mask(batch, "context_mask")
        target_embeddings = self.get_target_embeddings(target, target_masks)
        context_embeddings = self.get_context_embeddings(source, context_masks)
        predictions = self.predictor(
            context_embeddings, target_masks, batch[self.hparams.target_domain_key]
        )

        loss = self.loss(predictions, target_embeddings)
        return loss, None, None

    def get_predict_masks(self, batch_size, device):
        mask = torch.ones(self.hparams.encoder["num_patches"], dtype=bool, device=device)
        mask = rearrange(mask, "z y x -> (z y x)")
        mask = torch.argwhere(mask).squeeze()

        return repeat(mask, "t -> t b", b=batch_size)

    def extract_embeddings(self, tensor):
        # mean across patches, no cls token to remove
        return tensor.mean(axis=1).detach().cpu().numpy()

    def predict_step(self, batch, batch_idx):
        source = batch[self.hparams.source_key]
        source = self.remove_first_dim(source)

        embeddings = self.encoder(source)
        if self.hparams.predict_target:
            # use predictor to predict each patch
            target_masks = self.get_predict_masks(source.shape[0], device=source.device)
            # predict target embeddings from source embeddings
            embeddings = self.predictor(
                embeddings, target_masks, batch[self.hparams.target_domain_key]
            )
        return self.extract_embeddings(embeddings), source.meta

    def test_step(self, batch, batch_idx):
        source = batch[self.hparams.source_key]
        source = self.remove_first_dim(source)
        target = batch[self.hparams.target_key]
        target = self.remove_first_dim(target)

        # use predictor to predict each patch
        target_masks = self.get_predict_masks(source.shape[0], device=source.device)

        # mean across patches, no cls token to remove
        source_embeddings = self.encoder(source)
        # predict target embeddings from source embeddings
        pred_target_embeddings = self.predictor(
            source_embeddings, target_masks, batch[self.hparams.target_domain_key]
        )
        # get target embeddings
        target_embeddings = self.encoder(target)

        loss = self.loss(pred_target_embeddings, target_embeddings)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
