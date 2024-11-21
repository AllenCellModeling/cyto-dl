try:
    from .instance_seg import (
        InstanceSegCluster,
        InstanceSegLoss,
        InstanceSegPreprocessd,
    )
except (ModuleNotFoundError, ImportError):
    InstanceSegCluster = None
    InstanceSegLoss = None
    InstanceSegPreprocessd = None

from .metatensor_utils import metatensor_batch_to_tensor
from .postprocessing import ActThreshLabel, DictToIm, detach
