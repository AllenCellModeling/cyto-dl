from .adversarial_loss import AdversarialLoss
from .chamfer_loss import ChamferLoss
from .continuous_bernoulli import CBLogLoss
from .cosine_loss import CosineLoss
from .gan_loss import GANLoss, Pix2PixHD
from .gaussian_nll_loss import GaussianNLLLoss
from .threshold_loss import ThresholdLoss
from .vic_reg import VICRegLoss
from .weibull import WeibullLogLoss
from .weighted_mse_loss import WeightedMSELoss

try:
    from .spharm_loss import SpharmLoss
except (ModuleNotFoundError, ImportError):
    SpharmLoss = None

try:
    from .geomloss import GeomLoss
except (ModuleNotFoundError, ImportError):
    GeomLoss = None
