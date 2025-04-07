import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


class GradientLoggingCallback(Callback):
    def __init__(self, grouping_level: int = 3):
        super().__init__()
        self.grouping_level = grouping_level

    def _get_group(self, name):
        if self.grouping_level == -1:
            return name
        return ".".join(name.split(".")[: self.grouping_level])

    def on_train_epoch_end(self, trainer, pl_module):
        # Initialize a dictionary to store the average norms
        group_names = {self._get_group(name) for name, _ in pl_module.named_parameters()}

        norms = {name: [] for name in group_names}

        # Iterate over the model parameters
        for name, parameter in pl_module.named_parameters():
            # Check if the parameter has a norm
            if parameter.grad is not None:
                group_name = self._get_group(name)
                # Store the norm in the dictionary
                norms[group_name].append(torch.norm(parameter.grad).item())

        # Calculate the average norm for each group
        average_norms = {name: np.mean(norms[name]) for name in norms if len(norms[name]) > 0}
        # Log the average norms
        trainer.logger.log_metrics(average_norms, step=trainer.global_step)
