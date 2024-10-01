# goal: allow the API client to interact with CytoDLModel without
# worrying about the actual values being used in config files
from enum import Enum

import skimage


class ExperimentType(Enum):
    GAN = "gan"
    INSTANCE_SEG = "instance_seg"
    LABEL_FREE = "labelfree"
    SEGMENTATION_PLUGIN = "segmentation_plugin"
    SEGMENTATION = "segmentation"
    SEGMENTATION_ARRAY = "segmentation_array"


class HardwareType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    # other hardware types available, but require more complicated config


class PatchSize(Enum):
    """Patch size for training, and their respective patch shapes."""

    SMALL = [16, 32, 32]
    MEDIUM = [16, 64, 64]
    LARGE = [16, 128, 128]


# importing skimage takes a while.
# could speed it up by hardcoding this enum, but would have to update
# if skimage adds/deletes threshold filters

AutoThresholdMethod = Enum(
    "AutoThresholdMethod",
    {
        func_name.split("threshold_")[-1].upper(): func_name
        for func_name in filter(lambda x: x.startswith("threshold_"), dir(skimage.filters))
    },
)
