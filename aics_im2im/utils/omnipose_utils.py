# all code modified from https://github.com/kevinjohncutler/omnipose/blob/main/omnipose/core.py
from monai.data import MetaTensor
from monai.transforms import Transform
from omnipose.core import labels_to_flows, compute_masks
from omnipose.utils import get_boundary
from cellpose_omni.core import (
    WeightedLoss,
    ArcCosDotLoss,
    DerivativeLoss,
    NormLoss,
    DivergenceLoss,
)
import torch
import numpy as np
import edt
from skimage.filters import gaussian
from scipy.ndimage import laplace


def get_boundary_laplace(mask):
    """
    this gets an exterior boundary, instead of interior boundary like from omnipose
    however, it maintains boundaries between touching objects.
    """
    return laplace(mask) > 0


class OmniposePreprocessd(Transform):
    def __init__(self, label_key):
        super().__init__()
        self.label_key = label_key

    def __call__(self, image_dict):
        im = image_dict[self.label_key]
        numpy_im = im.numpy()
        im = im.as_tensor() if isinstance(im, MetaTensor) else im
        flows = labels_to_flows(numpy_im, use_gpu=True, device=im.device, dim=3)[0]
        # no boundary between merged nuclei, definitely change this
        boundary = get_boundary_laplace(numpy_im)
        bg_edt = edt.edt(numpy_im.squeeze() < 0.5, black_border=True)
        ## ARBITRARY, avoids complication from diameter calculation in omnipose
        cutoff = 20
        boundary_weighted_mask = (
            gaussian(1 - np.clip(bg_edt, 0, cutoff) / cutoff, 1) + 0.5
        )
        # back to CZYX
        boundary_weighted_mask = np.expand_dims(boundary_weighted_mask, 0)
        image_dict[self.label_key] = np.concatenate(
            [boundary, boundary_weighted_mask, flows], 0
        )
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

        return loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8


def OmniposeClustering(im):
    im = im.detach().cpu().numpy()
    flow = im[:3]
    dist = im[3]
    bd = im[4]
    mask, p, tr, bounds = compute_masks(
        flow,
        dist,
        bd,
        nclasses=4,
        dim=3,
    )
    return mask
