from .base_vae import BaseVAE
from .image_canon_vae import ImageCanonicalVAE
from .image_vae import ImageVAE
from .latent_loss_vae import LatentLossVAE
from .o2_spharm_vae.o2_spharm_vae import O2SpharmVAE
from .tabular_vae import TabularVAE
from .model_vae import ModelVAE

# compartmentalize imports so that only the relevant packages
# need to be installed
#
try:  # noqa: FURB107
    from .o2_spharm_vae.o2_spharm_vae import O2SpharmVAE
except ModuleNotFoundError:
    pass

# try:  # noqa: FURB107
from .point_cloud_vae import PointCloudVAE
# except ModuleNotFoundError:
#     pass
