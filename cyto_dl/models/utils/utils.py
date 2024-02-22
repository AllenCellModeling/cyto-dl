import numpy as np


def find_indices(lst, vals):
    arr = np.array(lst)

    sets = []
    for i, val in enumerate(vals):
        indices = set(np.where(arr == val)[0] - i)
        sets.append(indices)
    return np.asarray(list(set.intersection(*sets)), dtype=int)
