from contextlib import suppress

import pytest
import torch
from torchvision.models import get_model, list_models

from cyto_dl.nn import TorchVisionWrapper

# Filter the models to include only those that can be successfully loaded
valid_models = []
for model_name in list_models():
    with suppress(Exception):
        TorchVisionWrapper(get_model(model_name, weights=None))
        valid_models.append(model_name)
print(valid_models)


@pytest.mark.parametrize("model_name", valid_models)
def test_torchvision_wrapper(model_name):
    model = get_model(model_name, weights=None)
    wrapper = TorchVisionWrapper(model)
    img_size = (
        (1, 1, 448, 448)
        if "vit" not in model_name and "mc3" not in model_name
        else (1, 1, 224, 224)
    )
    img = torch.rand(*img_size)

    try:
        wrapper(img)
    except Exception as e:
        pytest.fail(f"Model {model_name} failed with error: {e}")
