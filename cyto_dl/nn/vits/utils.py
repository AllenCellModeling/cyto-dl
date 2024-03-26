import torch
from einops import repeat


def take_indexes(sequences, indexes):
    return torch.gather(
        sequences, 0, repeat(indexes.to(sequences.device), "t b -> t b c", c=sequences.shape[-1])
    )
