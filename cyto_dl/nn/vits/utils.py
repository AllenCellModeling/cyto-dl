import torch
from einops import repeat


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1]))


def random_indexes(size: int, device):
    forward_indexes = torch.randperm(size, device=device, dtype=torch.long)
    backward_indexes = torch.argsort(forward_indexes)
    return forward_indexes, backward_indexes
