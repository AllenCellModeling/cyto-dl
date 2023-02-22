# all code modified from https://github.com/kevinjohncutler/omnipose/blob/main/omnipose/core.py
import warnings

import dask
import edt
import numpy as np
import torch
import tqdm
from cellpose_omni.core import (
    ArcCosDotLoss,
    DerivativeLoss,
    DivergenceLoss,
    NormLoss,
    WeightedLoss,
)
from monai.data import MetaTensor
from monai.transforms import Transform
from omegaconf import ListConfig
from omnipose.core import compute_masks, diameters, masks_to_flows
from scipy.ndimage import find_objects
from scipy.spatial import ConvexHull
from skimage.filters import apply_hysteresis_threshold, gaussian
from skimage.measure import label
from skimage.morphology import binary_dilation, remove_small_holes
from skimage.segmentation import expand_labels
from skimage.transform import rescale, resize


class OmniposePreprocessd(Transform):
    def __init__(self, label_keys, dim):
        super().__init__()
        self.label_keys = (
            label_keys if isinstance(label_keys, (list, ListConfig)) else [label_keys]
        )
        self.dim = dim

    def __call__(self, image_dict):
        warnings.warn(
            "OmniPose preprocessing is slow, consider setting `persist_cache: True` in your experiment config"
        )
        for key in tqdm.tqdm(self.label_keys):
            im = image_dict[key]
            im = im.as_tensor() if isinstance(im, MetaTensor) else im
            numpy_im = im.numpy().squeeze()

            out_im = np.zeros([5 + self.dim] + list(numpy_im.shape))
            (
                instance_seg,
                rough_distance,
                boundaries,
                smooth_distance,
                flows,
            ) = masks_to_flows(numpy_im, omni=True, dim=self.dim, use_gpu=True, device=im.device)
            cutoff = diameters(instance_seg, rough_distance) / 2
            smooth_distance[rough_distance <= 0] = -cutoff

            bg_edt = edt.edt(numpy_im < 0.5, black_border=True)
            boundary_weighted_mask = gaussian(1 - np.clip(bg_edt, 0, cutoff) / cutoff, 1) + 0.5
            out_im[0] = boundaries
            out_im[1] = boundary_weighted_mask
            out_im[2] = instance_seg
            out_im[3] = rough_distance
            out_im[4 : 4 + self.dim] = flows * 5.0  # weighted for loss function?
            out_im[4 + self.dim] = smooth_distance
            image_dict[key] = out_im
        return image_dict


class OmniposeLoss:
    def __init__(self, dim):
        self.dim = dim
        self.weighted_flow_MSE = WeightedLoss()
        self.angular_flow_loss = ArcCosDotLoss()
        self.DerivativeLoss = DerivativeLoss()
        self.boundary_seg_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.NormLoss = NormLoss()
        self.distance_field_mse = WeightedLoss()
        self.criterion11 = DerivativeLoss()
        self.criterion16 = DivergenceLoss()

    def __call__(self, y, lbl):
        """Loss function for Omnipose.
        Parameters
        --------------
        lbl: ND-array, float
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]
            lbl[:,0] boundary field
            lbl[:,1] boundary-emphasized weights
            lbl[:,2] cell masks
            lbl[:,3] distance field
            lbl[:,4:6] flow components
            lbl[:,7] smooth distance field


        y:  ND-tensor, float
            network predictions, with dimension D, these are:
            y[:,:D] flow field components at 0,1,...,D-1
            y[:,D] distance fields at D
            y[:,D+1] boundary fields at D+1

        """
        boundary = lbl[:, 0]
        w = lbl[:, 1]
        cellmask = (lbl[:, 2] > 0).bool()  # acts as a mask now, not output

        # calculat loss on entire patch if no cells present - this helps
        # remove background artifacts
        for img_id in range(cellmask.shape[0]):
            if torch.sum(cellmask[img_id]) == 0:
                cellmask[img_id] = True
        veci = lbl[:, -(self.dim + 1) : -1]
        dist = lbl[:, -1]  # now distance transform replaces probability

        # prediction
        flow = y[:, : self.dim]  # 0,1,...self.dim-1
        dt = y[:, self.dim]
        bd = y[:, self.dim + 1]
        a = 10.0

        # stacked versions for weighting vector fields with scalars
        wt = torch.stack([w] * self.dim, dim=1)
        ct = torch.stack([cellmask] * self.dim, dim=1)

        # luckily, torch.gradient did exist after all and derivative loss was easy to implement. Could also fix divergenceloss, but I have not been using it.
        # the rest seem good to go.

        loss1 = 10.0 * self.weighted_flow_MSE(flow, veci, wt)  # weighted MSE
        loss2 = self.angular_flow_loss(flow, veci, w, cellmask)  # ArcCosDotLoss
        loss3 = self.DerivativeLoss(flow, veci, wt, ct) / a  # DerivativeLoss
        loss4 = 2.0 * self.boundary_seg_loss(bd, boundary)  # BCElogits
        loss5 = 2.0 * self.NormLoss(flow, veci, w, cellmask)  # loss on norm
        loss6 = 2.0 * self.distance_field_mse(dt, dist, w)  # weighted MSE
        loss7 = (
            self.criterion11(
                dt.unsqueeze(1),
                dist.unsqueeze(1),
                w.unsqueeze(1),
                cellmask.unsqueeze(1),
            )
            / a
        )
        loss8 = self.criterion16(flow, veci, cellmask)  # divergence loss
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
        return loss


# setting a high flow threshold avoids erroneous removal of masks that are fine.
# debugging whether this is a training issue...
class OmniposeClustering:
    """Run clustering on downsampled version of flows, then use original resolution distance field
    to mask instance segmentations."""

    def __init__(
        self,
        mask_threshold=0,
        rescale_factor=0.25,
        min_object_size=100,
        hole_size=0,
        flow_threshold=1e8,
        spatial_dim=3,
        boundary_seg=True,
        naive_label=False,
    ):

        self.mask_threshold = mask_threshold
        self.rescale_factor = rescale_factor
        self.min_object_size = min_object_size
        self.hole_size = hole_size
        self.flow_threshold = flow_threshold
        self.spatial_dim = spatial_dim
        self.boundary_seg = boundary_seg
        self.naive_label = naive_label
        self.clustering_function = self.do_naive_labeling if naive_label else self.get_mask

    def rescale_instance(self, im, seg):
        seg = resize(seg, im.shape, order=0, anti_aliasing=False, preserve_range=True)
        seg = expand_labels(seg, distance=3)
        seg[im == 0] = 0
        return seg

    def get_mask(self, flow, dist, device):
        flow = rescale(
            flow,
            [1] + [self.rescale_factor] * self.spatial_dim,
            order=3,
            preserve_range=True,
            anti_aliasing=False,
        )
        rescale_dist = rescale(
            dist, self.rescale_factor, order=3, preserve_range=True, anti_aliasing=False
        )
        mask, p, tr, bounds = compute_masks(
            flow,
            rescale_dist,
            nclasses=4,
            dim=self.spatial_dim,
            use_gpu=True,
            device=device,
            min_size=self.min_object_size,
            flow_threshold=self.flow_threshold,
            boundary_seg=self.boundary_seg,
            mask_threshold=self.mask_threshold,
        )
        mask = self.rescale_instance(dist > self.mask_threshold, mask)
        return mask

    def pad_slice(self, s, padding, constraints):
        new_slice = [slice(None, None, None)]
        for slice_part, c in zip(s, constraints):
            start = max(0, slice_part.start - padding)
            stop = min(c, slice_part.stop + padding)
            new_slice.append(slice(start, stop, None))
        return new_slice

    @dask.delayed
    def get_separated_masks(self, flow_crop, mask_crop, dist_crop, ratio_threshold, device, crop):
        if np.any(np.asarray(mask_crop.shape) < self.spatial_dim + 2):
            return
        area = np.sum(mask_crop)
        if area < self.min_object_size:
            return
        c_hull = ConvexHull(np.asarray(list(zip(*np.where(mask_crop))))).volume
        mask_crop = binary_dilation(mask_crop)
        if c_hull / area > ratio_threshold:
            flow_crop[:, ~mask_crop] = 0
            dist_crop[~mask_crop] = dist_crop.min()
            mask = self.get_mask(flow_crop, dist_crop, device)
            return {"slice": tuple(crop[1:]), "mask": mask}
        return {
            "slice": tuple(crop[1:]),
            "mask": remove_small_holes(
                mask_crop > 0, area_threshold=self.hole_size, connectivity=self.spatial_dim
            ),
        }

    def do_naive_labeling(self, flow, dist, device):
        """label thresholded distance transform to get objects, then run clustering only on objects
        that seem to be merged segmentations.

        Useful for well-separated, round objects like nuclei
        """
        cellmask = apply_hysteresis_threshold(
            dist, low=self.mask_threshold - 1, high=self.mask_threshold
        )
        naive_labeling = label(cellmask)
        out_image = np.zeros_like(naive_labeling, dtype=np.uint16)
        regions = find_objects(naive_labeling)
        results = []
        for val, region in enumerate(regions, start=1):
            padded_crop = self.pad_slice(region, 5, naive_labeling.shape)
            results.append(
                self.get_separated_masks(
                    flow[tuple(padded_crop)],
                    naive_labeling[tuple(padded_crop[1:])] == val,
                    dist[tuple(padded_crop[1:])],
                    1.2,
                    device,
                    padded_crop,
                )
            )
        results = dask.compute(*results)
        highest_cell_idx = 0
        for r in results:
            if r is None:
                continue
            mask = r["mask"].astype(np.uint16)
            mask[mask > 0] += highest_cell_idx
            out_image[r["slice"]] += mask
            highest_cell_idx += np.max(r["mask"])
        return out_image

    def __call__(self, im):
        device = im.device
        im = im.detach().cpu().numpy()
        flow = im[: self.spatial_dim]
        dist = im[self.spatial_dim]
        mask = self.clustering_function(flow, dist, device)
        return mask