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

from .postprocessing import ActThreshLabel, DictToIm, detach
