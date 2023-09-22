from typing import List, Optional, Union

import torch
from lightning.pytorch.callbacks import Callback


class LayerFreeze(Callback):
    def __init__(
        self,
        contains: Optional[Union[str, List[str]]] = None,
        excludes: Optional[Union[str, List[str]]] = None,
    ):
        assert (
            contains is not None or excludes is not None
        ), "One of `contains` or `excludes` must be provided"
        self.contains = contains or []
        self.excludes = excludes or []

    def _filter(self, mod):
        for n, p in mod.named_parameters():
            requires_grad = True
            for contain in self.contains:
                if contain in n:
                    requires_grad = False
                    break
            for exclude in self.excludes:
                if exclude in n:
                    requires_grad = True
                    break
            if not requires_grad:
                print(f"\tFreezing layer {n}")
            p.requires_grad = requires_grad

    def setup(self, trainer, pl_module, stage):
        for module in dir(pl_module):
            if isinstance(getattr(pl_module, module), torch.nn.Module):
                print(f"Freezing layers in {module}")
                self._filter(getattr(pl_module, module))
