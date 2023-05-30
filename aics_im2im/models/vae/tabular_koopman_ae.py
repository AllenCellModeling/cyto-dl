from typing import Sequence

from aics_im2im.nn import MLP

from .koopman_ae import KoopmanAE


class TabularKoopmanAE(KoopmanAE):
    def __init__(
        self,
        *,
        x_dim: int,
        latent_dim: int,
        hidden_layers: Sequence[int],
        **base_kwargs,
    ):
        """Instantiate an Image Koopman Autoencoder model.

        Parameters
        ----------
        latent_dim: int
            Bottleneck size
        x_label: Optional[str] = None
        rank: int = 0
            The rank of the tensor decomposition
        **base_kwargs:
            Additional arguments passed to BaseModel
        """

        encoder = MLP(
            x_dim,
            latent_dim,
            hidden_layers=hidden_layers,
        )

        decoder = MLP(
            latent_dim,
            x_dim,
            hidden_layers=hidden_layers,
        )

        super().__init__(encoder=encoder, decoder=decoder, latent_dim=latent_dim, **base_kwargs)
