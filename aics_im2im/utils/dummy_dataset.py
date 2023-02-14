import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 10000, **shapes):
        """
        Args:
            num_samples: how many samples to use in this dataset
            **shapes: kwargs, where each key will become a batch key and each
                      value is the shape of the corresponding batch elements
        """
        super().__init__()
        self.shapes = shapes

        if num_samples < 1:
            raise ValueError("Provide an argument greater than 0 for `num_samples`")

        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return {k: torch.randn(*shape) for k, shape in self.shapes.items()}
