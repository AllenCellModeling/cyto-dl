from typing import List, Optional, Union

import torch
from lightning.pytorch.callbacks import Callback


class LayerFreeze(Callback):
    def __init__(
        self,
        modules: Union[str, List[str]] = [],
        contains: Optional[Union[str, List[str]]] = None,
        excludes: Optional[Union[str, List[str]]] = None,
    ):
        """
        Parameters
        ----------
        modules: Union[str, List[str]]
            List of modules to search within
        contains: Optional[Union[str, List[str]]]
            List of strings that must be contained in the layer name to freeze
        excludes: Optional[Union[str, List[str]]]
            List of strings that must not be contained in the layer name to freeze
        """
        assert (
            contains is not None or excludes is not None
        ), "One of `contains` or `excludes` must be provided"
        self.modules = modules
        contains = contains or []
        self.contains = [contains] if isinstance(contains, str) else contains
        excludes = excludes or []
        self.excludes = [excludes] if isinstance(excludes, str) else excludes

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
        for module in self.modules:
            if isinstance(getattr(pl_module, module), torch.nn.Module):
                print(f"Freezing layers in {module}")
                self._filter(getattr(pl_module, module))
