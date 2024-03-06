<<<<<<< HEAD
from .head import BaseHead, ConvProjectionLayer, GANHead, ResBlocksHead, PointCloudHead
=======
from .head import BaseHead, GANHead, GANHead_resize, ResBlocksHead
>>>>>>> 0c2b702edcfe82a83595330b0b60433cc40139ae
from .hr_skip import HRSkip
from .losses import (
    AdversarialLoss,
    CBLogLoss,
    CosineLoss,
    GaussianNLLLoss,
    WeibullLogLoss,
)
from .mlp import MLP
from .mlp_onehot import MLPOHot
from .res_unit import ResidualUnit
from .spatial_transformer import STN
