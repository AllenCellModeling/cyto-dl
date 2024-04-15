import torch
from einops import repeat


def take_indexes(sequences, indexes):
    # always gather across tokens dimension
    if len(sequences.shape) == 4:
        return torch.gather(
            sequences, 1, repeat(indexes.to(sequences.device), "t b -> n t b c", n = sequences.shape[0], c=sequences.shape[-1])
        )
    elif len(sequences.shape) == 3:
        return torch.gather(
            sequences, 0, repeat(indexes.to(sequences.device), "t b -> t b c", c=sequences.shape[-1])
        )
