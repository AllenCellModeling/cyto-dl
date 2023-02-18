# all code modified from https://github.com/kevinjohncutler/omnipose/blob/main/omnipose/core.py
import warnings

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
from omnipose.core import compute_masks, diameters, masks_to_flows
from skimage.filters import gaussian
from skimage.segmentation import expand_labels
from skimage.transform import rescale, resize


class OmniposePreprocessd(Transform):
    def __init__(self, label_key, dim):
        super().__init__()
        self.label_key = label_key
        self.dim = dim

    def __call__(self, image_dict):
        warnings.warn(
            "OmniPose preprocessing is slow, consider setting `persist_cache: True` in your experiment config"
        )
        for _ in tqdm.tqdm(range(1)):
            im = image_dict[self.label_key]
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
            image_dict[self.label_key] = out_im
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


def rescale_instance(im, seg):
    seg = resize(seg, im.shape, order=0, anti_aliasing=False, preserve_range=True)
    seg = expand_labels(seg, distance=3)
    seg[im == 0] = 0
    return seg


# setting a high flow threshold avoids erroneous removal of masks that are fine.
# debugging whether this is a training issue...
def OmniposeClustering(
    im,
    mask_threshold=0,
    rescale_factor=0.25,
    min_object_size=100,
    flow_threshold=1e8,
    spatial_dim=3,
):
    """Run clustering on downsampled version of flows, then use original resolution distance field
    to mask instance segmentations."""
    device = im.device
    im = im.detach().cpu().numpy()
    flow = im[:spatial_dim]
    dist = im[spatial_dim]
    # bd = im[spatial_dim+1]

    mask, p, tr, bounds = compute_masks(
        rescale(
            flow,
            [1] + [rescale_factor] * spatial_dim,
            order=3,
            preserve_range=True,
            anti_aliasing=False,
        ),
        rescale(dist, rescale_factor, order=3, preserve_range=True, anti_aliasing=False),
        # bd,
        nclasses=4,
        dim=spatial_dim,
        use_gpu=True,
        device=device,
        min_size=min_object_size,
        flow_threshold=flow_threshold,
        boundary_seg=True,
        mask_threshold=mask_threshold,
    )
    mask = rescale_instance(dist > mask_threshold, mask)
    return mask
