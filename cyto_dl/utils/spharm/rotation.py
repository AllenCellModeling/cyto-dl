import itertools
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss


def flatten(lst):
    return list(itertools.chain.from_iterable(lst))


def get_indices(columns, max_band, prefix=""):
    # the spherical harmonics in our data have the form
    # {prefix}{band}M{m}{sign}
    # where `m` varies from 0 to `band`,
    # and `sign` is either "C" or "S" (for cosine and sine), and
    # which correspond to positive and negative m values, respectively
    pattern = re.compile(prefix + r".*L([0-9]*)M([0-9]*)(S|C)")

    m_groups = [list((None,)) * 2 * m for m in range(1, max_band + 1)][::-1]
    zero_indices = []

    for ix, col in enumerate(columns):
        # parse the relevant values from the column string
        band, m, sign = pattern.match(col).groups()

        band = int(band)
        m = int(m)
        sign = -1 if sign == "S" else 1

        if m == 0:
            zero_indices.append(ix)
            continue

        # get indices for each m group ordered by band. each band has a pair.
        # each pair in each m group gets rotated by the same angle
        m_groups[m - 1][2 * (band - m) + (sign > 0)] = ix

    return get_band_indices(columns, max_band, prefix, flat=True), m_groups


def build_rot_matrices(angles):
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


def expand_rot_matrices(rot, max_band):
    rot_matrices = []
    rot = torch.stack(
        (
            torch.stack((rot[:, 0], -rot[:, 1]), dim=-1),
            torch.stack((rot[:, 1], rot[:, 0]), dim=-1),
        ),
        dim=-1,
    )

    _rot = torch.eye(2, 2).repeat(rot.shape[0], 1, 1).to(rot.device)

    for m in range(1, max_band + 1):
        _rot = torch.matmul(_rot, rot)

        rot_matrices.append(
            _rot.expand((max_band - (m - 1)), _rot.shape[0], 2, 2).permute(1, 0, 2, 3)
        )

    return torch.concatenate(rot_matrices, dim=1)


def _rotate_spharm(input, rot_matrices):
    b, m = input.shape[:2]

    return torch.matmul(
        rot_matrices.view(b * m, 2, 2),
        input.view(b * m, 2, 1),
    ).view(b, m * 2)


def rotate_spharm(input, rot, paired_indices, max_band):
    flat_idx = flatten(paired_indices)
    num_groups = len(flat_idx) // 2

    input[:, flat_idx] = _rotate_spharm(
        input[:, flat_idx].view(input.shape[0], num_groups, 2),
        expand_rot_matrices(rot, max_band),
    )

    return input


def flip_spharm(input, paired_indices, flips=-1):
    for m_group in paired_indices:
        neg = [true_ix for ix, true_ix in enumerate(m_group) if ix % 2 == 0]
        input[:, neg] = flips * input[:, neg]
    return input


def get_band_indices(columns, max_band, prefix="", flat=False):
    """Get the tensor indices for each band, based on the column order of the batch loader (given
    by `columns`, assuming that it is in the same order).

    this is passed to `rotate_spharm` later, to rotate the spherical harmonics around the z axis
    """

    # the spherical harmonics in our data have the form
    # {prefix}{band}M{m}{sign}
    # where `m` varies from 0 to `band`,
    # and `sign` is either "C" or "S" (for cosine and sine), and
    # which correspond to positive and negative m values, respectively
    pattern = re.compile(prefix + r".*L([0-9]*)M([0-9]*)(S|C)")
    bands = []
    for band in range(0, max_band + 1):
        # each band has 2 * band + 1 elements, but here we exclude m=0
        # because it remains unchanged under rotations around the Z axis
        band_size = 2 * band + 1
        bands.append(list((None,)) * band_size)

    for ix, col in enumerate(columns):
        # parse the relevant values from the column string
        try:
            band, m, sign = pattern.match(col).groups()
        except AttributeError:
            continue

        band = int(band)
        m = int(m)
        sign = -1 if sign == "S" else 1

        # for this band, the element corresponding to `(sign)m` is at the index `ix`
        # of the tensor. we sum it by `band - (sign > 0)` such that the left most element is
        # at the list index 0.
        #
        # e.g., for band = 2, we have the correspondence:
        # [-2, -1, 0, 1, 2] -> [0, 1, 2, 3, 4]

        bands[band][(sign * m) + band] = ix

    if not flat:
        return bands
    return flatten(bands)
