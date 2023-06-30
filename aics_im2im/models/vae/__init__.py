from .base_vae import BaseVAE
from .image_vae import ImageVAE
from .latent_loss_vae import LatentLossVAE
from .o2_spharm_vae.o2_spharm_vae import O2SpharmVAE
from .so2_image_vae import SO2ImageVAE
from .tabular_vae import TabularVAE

# compartmentalize imports so that only the relevant packages
# need to be installed
#
try:  # noqa: FURB107
    from .o2_spharm_vae.o2_spharm_vae import O2SpharmVAE
except ModuleNotFoundError:
    pass

try:  # noqa: FURB107
    from .point_cloud_vae import PointCloudVAE
except ModuleNotFoundError:
    pass
