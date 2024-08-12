import torch.nn as nn

from cyto_dl.models.jepa import JEPABase


class IJEPA(JEPABase):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        predictor: nn.Module,
        x_key: str,
        save_dir: str = "./",
        momentum: float = 0.998,
        max_epochs: int = 100,
        **base_kwargs,
    ):
        """JEPA for self-supervised learning on 2D and 3D images.

        Parameters
        ----------
        encoder : nn.Module
            The encoder module used for feature extraction.
        predictor : nn.Module
            The predictor module used for generating predictions.
        x_key : str
            The key used to access the input data.
        momentum : float, optional
            The momentum value for the exponential moving average of the model weights (default is 0.998).
        max_epochs : int, optional
            The maximum number of training epochs (default is 100).
        **base_kwargs : dict
            Additional arguments passed to the BaseModel.
        """
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            x_key=x_key,
            save_dir=save_dir,
            momentum=momentum,
            max_epochs=max_epochs,
            **base_kwargs,
        )

    def model_step(self, stage, batch, batch_idx):
        self.update_teacher()
        input = batch[self.hparams.x_key]
        input = self.remove_first_dim(input)

        target_masks = self.get_mask(batch, "target_mask")
        context_masks = self.get_mask(batch, "context_mask")

        target_embeddings = self.get_target_embeddings(input, target_masks)
        context_embeddings = self.get_context_embeddings(input, context_masks)
        predictions = self.predictor(context_embeddings, target_masks)

        loss = self.loss(predictions, target_embeddings)
        return loss, None, None
