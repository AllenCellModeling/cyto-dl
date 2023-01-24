# MODIFIED FROM https://github.com/MMV-Lab/mmv_im2im/blob/main


import numpy as np
from typing import Union, Tuple, List
from numba import jit
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_fill_holes
from pathlib import Path
import torch
import torch.nn as nn
from torch.autograd import Variable
from itertools import filterfalse  # noqa F401
import torch.nn.functional as F
from monai.transforms import Transform
from monai.data.meta_tensor import MetaTensor


@jit(nopython=True)
def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D


class EmbedSegPreprocess(Transform):
    def __init__(
        self,
        input_key,
        output_key,
        anisotropy_factor=1,
        centroid_name="centroid",
        downsample=4,
    ):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.anisotropy_factor = anisotropy_factor
        self.centroid_name = centroid_name
        self.downsample = downsample

    def _get_centroid(self, image):
        ids = torch.unique(image)
        center_image = np.zeros(image.shape, dtype=bool)
        instance_downsampled = image[
            :, :, :: int(self.downsample), :: int(self.downsample)
        ]  # down sample in x and y
        for id in ids:
            c, z, y, x = np.where(instance_downsampled == id)
            if len(y) != 0 and len(x) != 0:
                if self.centroid_name == "centroid":
                    zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
                elif self.centroid_name == "approximate-medoid":
                    zm_temp, ym_temp, xm_temp = np.median(z), np.median(y), np.median(x)
                    imin = np.argmin(
                        (x - xm_temp) ** 2
                        + (y - ym_temp) ** 2
                        + (self.anisotropy_factor * (z - zm_temp)) ** 2
                    )
                    zm, ym, xm = z[imin], y[imin], x[imin]
                elif self.centroid_name == "medoid":
                    dist_matrix = pairwise_python(
                        np.vstack(
                            (
                                self.downsample * x,
                                self.downsample * y,
                                self.anisotropy_factor * z,
                            )
                        ).transpose()
                    )
                    imin = np.argmin(np.sum(dist_matrix, axis=0))
                    zm, ym, xm = z[imin], y[imin], x[imin]
                center_image[
                    c,
                    int(np.round(zm)),
                    int(np.round(self.downsample * ym)),
                    int(np.round(self.downsample * xm)),
                ] = True
        return torch.from_numpy(center_image)

    def __call__(self, image_dict):
        image_dict[self.output_key["class"]] = image_dict[self.input_key] > 0
        image_dict[self.output_key["centroid"]] = self._get_centroid(
            image_dict[self.input_key]
        )
        return image_dict


class EmbedSegConcatLabelsd(Transform):
    def __init__(self, input_keys, output_key):
        super().__init__()
        self.input_keys = input_keys
        self.output_key = output_key

    def __call__(self, image_dict):
        image_dict[self.output_key] = {}
        for key in self.input_keys:
            im = image_dict[key]
            im = im.as_tensor() if isinstance(im, MetaTensor) else im
            image_dict[self.output_key][key] = im
            del image_dict[key]
        return image_dict


class Cluster_3d:
    def __init__(
        self, grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x, one_hot=False
    ):

        xm = (
            torch.linspace(0, pixel_x, grid_x)
            .view(1, 1, 1, -1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        ym = (
            torch.linspace(0, pixel_y, grid_y)
            .view(1, 1, -1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        zm = (
            torch.linspace(0, pixel_z, grid_z)
            .view(1, -1, 1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        xyzm = torch.cat((xm, ym, zm), 0)

        self.xyzm = xyzm
        self.one_hot = one_hot
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.pixel_z = pixel_z

    def cluster_with_gt(
        self,
        prediction,
        image,
        n_sigma=1,
    ):
        self.xyzm = self.xyzm.type_as(prediction)
        depth, height, width = (
            prediction.size(1),
            prediction.size(2),
            prediction.size(3),
        )
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]  # 3 x d x h x w
        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w
        sigma = prediction[3 : 3 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(depth, height, width).short().type_as(prediction)
        unique_instances = image.unique()
        unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            mask = image.eq(id).view(1, depth, height, width)
            center = (
                spatial_emb[mask.expand_as(spatial_emb)]
                .view(3, -1)
                .mean(1)
                .view(3, 1, 1, 1)
            )  # 3 x 1 x 1 x 1
            s = (
                sigma[mask.expand_as(sigma)]
                .view(n_sigma, -1)
                .mean(1)
                .view(n_sigma, 1, 1, 1)
            )

            s = torch.exp(s * 10)  # n_sigma x 1 x 1
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            proposal = dist > 0.5
            instance_map[proposal] = id.item()  # TODO

        return instance_map

    def cluster(
        self,
        prediction,
        n_sigma=3,
        seed_thresh=0.5,
        min_mask_sum=128,
        min_unclustered_sum=0,
        min_object_size=36,
    ):
        self.xyzm = self.xyzm.type_as(prediction)

        depth, height, width = (
            prediction.size(1),
            prediction.size(2),
            prediction.size(3),
        )
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w

        sigma = prediction[3 : 3 + n_sigma]  # n_sigma x d x h x w
        seed_map = torch.sigmoid(
            prediction[3 + n_sigma : 3 + n_sigma + 1]
        )  # 1 x d x h x w
        instance_map = torch.zeros(depth, height, width).short()
        instances = []  # list

        count = 1
        mask = seed_map > 0.5
        if mask.sum() > min_mask_sum:
            # top level decision: only start creating instances, if there are atleast
            # 128 pixels in foreground!

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(
                n_sigma, -1
            )
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().type_as(prediction)
            instance_map_masked = torch.zeros(mask.sum()).short().type_as(prediction)

            while (
                unclustered.sum() > min_unclustered_sum
            ):  # stop when the seed candidates are less than 128
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < seed_thresh:
                    break
                center = spatial_emb_masked[:, seed : seed + 1]
                unclustered[seed] = 0

                s = torch.exp(sigma_masked[:, seed : seed + 1] * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0)
                )

                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if (
                        unclustered[proposal].sum().float() / proposal.sum().float()
                        > 0.5
                    ):
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(depth, height, width).short()
                        instance_mask[mask.squeeze().cpu()] = proposal.short().cpu()
                        count += 1
                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.short().cpu()

        return instance_map, instances


def generate_instance_clusters(
    pred: Union[np.ndarray, torch.Tensor],
    pixel_x: int = 1,
    pixel_y: int = 1,
    n_sigma: int = 2,
    seed_thresh: float = 0.5,
    min_mask_sum: int = 10,
    min_unclustered_sum: int = 10,
    min_object_size: int = 10,
    pixel_z: int = 1,
):
    ##########################################################
    # Warninging: Even passing in a mini-batch, only the first
    # one will be used.
    ##########################################################
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)

    if len(pred.shape) == 5:
        pred = pred[0]  # save time during training, only save 1st example in batch
    if len(pred.shape) == 4:  # C x Z x Y x X
        cluster = Cluster_3d(
            pred.shape[-3], pred.shape[-2], pred.shape[-1], pixel_z, pixel_y, pixel_x
        )
    else:
        raise ValueError("prediction needs to be 4D in cluster")

    instance_map, _ = cluster.cluster(
        pred,
        n_sigma=n_sigma,
        seed_thresh=seed_thresh,
        min_mask_sum=min_mask_sum,
        min_unclustered_sum=min_unclustered_sum,
        min_object_size=min_object_size,
    )

    return instance_map.cpu().numpy()


# adapted from https://github.com/juglab/EmbedSeg/tree/main/EmbedSeg/criterions


class SpatialEmbLoss_3d(nn.Module):
    def __init__(
        self,
        grid_z=32,
        grid_y=1024,
        grid_x=1024,
        pixel_z=1,
        pixel_y=1,
        pixel_x=1,
        n_sigma=3,
        foreground_weight=10,
        use_costmap=False,
        instance_key="GT",
        costmap_key="CM",
        label_key="CL",
        center_key="CE",
    ):
        super().__init__()

        print(
            f"Created spatial emb loss function with: n_sigma: {n_sigma},"
            f"foreground_weight: {foreground_weight}"
        )
        print("*************************")
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        xm = (
            torch.linspace(0, pixel_x, grid_x)
            .view(1, 1, 1, -1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        ym = (
            torch.linspace(0, pixel_y, grid_y)
            .view(1, 1, -1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        zm = (
            torch.linspace(0, pixel_z, grid_z)
            .view(1, -1, 1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        xyzm = torch.cat((xm, ym, zm), 0)

        self.register_buffer("xyzm", xyzm)
        self.use_costmap = use_costmap
        self.instance_key = instance_key
        self.costmap_key = costmap_key
        self.label_key = label_key
        self.center_key = center_key

    def forward(
        self,
        prediction,
        target,
        w_inst=1,
        w_var=10,
        w_seed=1,
    ):
        self.xyzm = self.xyzm.type_as(prediction)
        # instances B 1 Z Y X
        batch_size, depth, height, width = (
            prediction.size(0),
            prediction.size(2),
            prediction.size(3),
            prediction.size(4),
        )
        instances = target[self.instance_key].short()
        costmaps = target.get(self.costmap_key)
        labels = target[self.label_key].bool()
        center_images = target[self.center_key].bool()
        # weighted loss
        if self.use_costmap:
            # only need to adjust instances in this step, because for pixels
            # with zero weight, this step will ignore the corresponding instances
            instances_adjusted = instances * costmaps
        else:
            instances_adjusted = instances

        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width].contiguous()  # 3 x d x h x w

        loss = 0

        for b in range(0, batch_size):
            spatial_emb = torch.tanh(prediction[b, 0:3]) + xyzm_s  # 3 x d x h x w
            sigma = prediction[b, 3 : 3 + self.n_sigma]  # n_sigma x d x h x w
            seed_map = torch.sigmoid(
                prediction[b, 3 + self.n_sigma : 3 + self.n_sigma + 1]
            )  # 1 x d x h x w
            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            if self.use_costmap:
                costmap = costmaps[b]
            image = instances[b]  # without costmap adjustment
            label = labels[b]
            center_image = center_images[b]

            # use adjusted image to find all ids
            instance_ids = instances_adjusted[b].unique()
            instance_ids = instance_ids[instance_ids != 0]

            # regress bg to zero
            bg_mask = label == 0

            if bg_mask.sum() > 0:
                if self.use_costmap:
                    # adjust the cost here, because some of the background pixels might
                    # have zero weight
                    seed_loss += torch.sum(
                        costmap * torch.pow(seed_map[bg_mask] - 0, 2)
                    )
                else:
                    seed_loss += torch.sum(torch.pow(seed_map[bg_mask] - 0, 2))

            for id in instance_ids:

                # use the original image without costmap adjustment to fetch
                # image mask, since the costmap may partial cut some instances
                # and alter the ground truth only use the costmap to adjust the
                # loss values at the end
                in_mask = image.eq(id)  # 1 x d x h x w
                center_mask = in_mask & center_image

                if center_mask.sum().eq(1):
                    center = xyzm_s.masked_select(center_mask.expand_as(xyzm_s)).view(
                        3, 1, 1, 1
                    )
                else:
                    xyz_in = xyzm_s.masked_select(in_mask.expand_as(xyzm_s)).view(3, -1)
                    center = xyz_in.mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1 x 1

                # calculate sigma
                sigma_in = sigma[in_mask.expand_as(sigma)].view(
                    self.n_sigma, -1
                )  # 3 x N

                s = sigma_in.mean(1).view(self.n_sigma, 1, 1, 1)  # n_sigma x 1 x 1 x 1

                # calculate var loss before exp
                if self.use_costmap:
                    var_loss = var_loss + torch.mean(
                        costmap * torch.pow(sigma_in - s[..., 0, 0].detach(), 2)
                    )
                else:
                    var_loss = var_loss + torch.mean(
                        torch.pow(sigma_in - s[..., 0, 0].detach(), 2)
                    )

                s = torch.exp(s * 10)
                s = torch.nan_to_num(s)
                dist = torch.exp(
                    -1
                    * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True)
                )

                # apply lovasz-hinge loss
                # TODO: currently, if we assume the costmap is just to make some
                # instances on/off. this loss is still good. Otherwise, there might be
                # a little off.
                instance_loss = instance_loss + lovasz_hinge(dist * 2 - 1, in_mask)

                # seed loss
                if self.use_costmap:
                    seed_loss += self.foreground_weight * torch.sum(
                        costmap
                        * torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2)
                    )
                else:
                    seed_loss += self.foreground_weight * torch.sum(
                        torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2)
                    )

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            if self.use_costmap:
                seed_loss = seed_loss / costmap.sum()
            else:
                seed_loss = seed_loss / (depth * height * width)

            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss

        loss = loss / (b + 1)

        return loss + prediction.sum() * 0


def mean(l, ignore_nan=False, empty=0):  # noqa E741
    """
    nanmean compatible with generators.
    """
    l = iter(l)  # noqa E741
    if ignore_nan:
        l = filterfalse(np.isnan, l)  # noqa E741
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc

    return acc / n


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)  # noqa W605
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(
                *flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts.float() - gt_sorted.float().cumsum(0)
    # union = gts.float() + (1 - gt_sorted).float().cumsum(0)
    union = gts.float() + (1 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)  # noqa W605
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels