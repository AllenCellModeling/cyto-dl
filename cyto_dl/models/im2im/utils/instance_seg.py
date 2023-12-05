from typing import Sequence, Union

import edt
import numpy as np
import torch

# implements a nan-ignoring convolution
from astropy.convolution import convolve
from monai.data import MetaTensor
from monai.losses import TverskyLoss
from monai.transforms import Flip, RandomizableTransform, Transform
from omegaconf import ListConfig
from scipy.ndimage import find_objects
from scipy.spatial import KDTree
from skimage.filters import apply_hysteresis_threshold, gaussian
from skimage.measure import label
from skimage.morphology import (
    ball,
    dilation,
    disk,
    erosion,
    remove_small_objects,
    skeletonize,
)
from skimage.segmentation import find_boundaries

from cyto_dl.nn.losses.loss_wrapper import CMAP_loss


class InstanceSegPreprocessd(Transform):
    def __init__(
        self,
        label_keys: Union[Sequence[str], str],
        kernel_size: int = 3,
        thin: int = 5,
        dim: int = 3,
        anisotropy: float = 2.6,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        label_keys: Union[Sequence[str], str]
            Keys of instance segmentations in input dictionary to convert to InstanceSeg ground truth images.
        kernel_size: int=3
            Size of kernel for gaussian smoothing of flows
        thin: int=5
            Amount to thin to create psuedo-skeleton
        dim:int=3
            Spatial dimension of images
        anisotropy:float=2.6
            Anisotropy of images
        allow_missing_keys:bool=False
            Whether to raise error if key in `label_keys` is not present
        """
        super().__init__()
        self.label_keys = (
            label_keys if isinstance(label_keys, (list, ListConfig)) else [label_keys]
        )
        self.dim = dim
        self.allow_missing_keys = allow_missing_keys
        self.kernel_size = kernel_size
        self.anisotropy = torch.as_tensor([anisotropy if dim == 3 else 1] + [1] * (dim - 1))
        self.thin = thin

    def shrink(self, im):
        """Topology-preserving thinning of a binary image."""
        skel = np.zeros_like(im)
        regions = find_objects(im)
        for lab, coords in enumerate(regions, start=1):
            if coords is None:
                continue
            skel[coords] += self.topology_preserving_thinning(im[coords] == lab)
        return skel * im  # relabel shrunk object

    def skeleton_tall(self, img, max_label):
        """Skeletonize 3d image with increased thickness in z."""
        if max_label == 0 or self.dim == 2:
            return skeletonize(img)
        tall_skeleton = np.stack([skeletonize(np.max(img, 0))] * img.shape[0])
        return tall_skeleton

    def label_slice(self, img):
        out = np.zeros_like(img, dtype=np.int16)
        for z in range(img.shape[0]):
            lab = label(img[z])
            lab[lab > 0] += np.max(out)
            out[z] = lab
        return out

    def topology_preserving_thinning(self, bw, min_size=100):
        """Topology-preserving thinning of a binary image.

        Use skeleton to bridge gaps created by erosion.
        """
        selem = ball(self.thin)[:: int(self.anisotropy[0])] if self.dim == 3 else disk(self.thin)
        eroded = erosion(bw, selem)
        # only want to preserve connections between significantly-sized objects

        eroded = remove_small_objects(eroded, min_size)
        eroded, max_label = label(eroded, return_num=True)

        # single object is preserved by erosion
        if max_label == 1:
            return eroded

        skel = self.skeleton_tall(bw, max_label)
        if max_label == 0:
            return skel

        # if erosion separates object into multiple pieces, use skeleton to bridge those pieces into single object
        # 1. isolate pieces of skeleton that are outside of eroded objects (i.e. could bridge between objects)
        skel[eroded != 0] = 0
        skel = self.label_slice(skel) if self.dim == 3 else label(skel)

        for i in np.unique(skel)[1:]:
            # 3. find number of non-background objects overlapped by piece of skeleton, add back in pieces that overlap multiple obj
            dilation_selem = np.expand_dims(disk(3), 0)
            dilated_skel = dilation(skel == i, dilation_selem)
            n_obj_masked = np.sum(np.unique(eroded[dilated_skel]) > 0)
            if n_obj_masked > 1:
                eroded += dilated_skel
        # make sure dilated skeleton is within object bounds by 1 pix so vectors can point to it
        one_erode = erosion(bw)
        eroded[one_erode == 0] = 0
        return eroded > 0

    def _get_point_embeddings(self, object_points, skeleton_points):
        """Finds closest skeleton point to each object point using KDTree."""
        tree = KDTree(skeleton_points)
        dist, idx = tree.query(object_points)
        return torch.from_numpy(tree.data[idx]).T.float()

    def smooth_embedding(self, embedding):
        """Smooths embedding by convolving with a mean kernel, excluding non-object pixels."""
        kernel = np.ones([self.kernel_size] * self.dim) / self.kernel_size**self.dim
        nan_embed = embedding.clone()
        nan_embed[nan_embed == 0] = torch.nan
        for i in range(embedding.shape[0]):
            conv_embed = convolve(nan_embed[i].numpy(), kernel, boundary="extend")
            conv_embed[torch.isnan(nan_embed[i])] = 0
            embedding[i] = torch.from_numpy(conv_embed)
        return embedding

    def embed_from_skel(self, skel, iseg):
        """Find per-pixel embedding vector to closest point on skeleton."""
        iseg[skel != 0] = 0

        # 3ZYX vector field for 3d, 2YX for 2d
        embed = torch.zeros([self.dim] + [iseg.shape[i] for i in range(self.dim)])

        # propagate labels to boundaries
        skel_boundary = torch.from_numpy(find_boundaries(skel.numpy(), mode="inner")) * skel
        for i in np.unique(iseg)[1:]:
            object_mask = iseg.eq(i)
            # distances should take into account z anisotropy
            object_points = object_mask.nonzero().mul(self.anisotropy)
            skel_points = skel_boundary.eq(i).nonzero().mul(self.anisotropy)
            if skel_points.numel() == 0:
                continue
            point_embeddings = self._get_point_embeddings(object_points, skel_points)
            embed[:, object_mask] = point_embeddings
        # smooth sharp transitions from spatial embedding
        embed = self.smooth_embedding(embed)

        # turn spatial embedding into offset vector by subtracting pixel coordinates
        anisotropic_shape = torch.as_tensor(iseg.shape).mul(self.anisotropy)
        coordinates = torch.stack(
            torch.meshgrid(
                *[
                    torch.linspace(0, anisotropic_shape[i] - 1, iseg.shape[i])
                    for i in range(self.dim)
                ]
            )
        )
        embed_pts = embed.ne(0)
        embed[embed_pts] -= coordinates[embed_pts]
        return embed

    def _get_object_contacts(self, img):
        """Find pixels that separate touching objects."""
        regions = find_objects(img.astype(int))
        outer_bounds = np.zeros_like(img)
        for lab, coords in enumerate(regions, start=1):
            bounds = find_boundaries(img[coords] == lab)
            outer_bounds[coords] += bounds
        outer_bounds = outer_bounds > 1
        return (outer_bounds * 10).squeeze()

    def _get_cmap(self, skel_edt, im):
        """Create costmap to increase loss in boundary areas."""
        points_with_vecs = im.clone().squeeze()
        points_with_vecs[skel_edt > 0] = 0
        # emphasize very thin areas
        add_in_thin = np.logical_and(skel_edt > 0, skel_edt < 3)
        # emphasize areas where vector field is nonzero
        points_with_vecs = np.logical_or(points_with_vecs, add_in_thin)
        sigma = torch.as_tensor([2] * self.dim) / self.anisotropy
        sigma = torch.max(sigma, torch.ones(self.dim)).numpy()
        cmap = gaussian(points_with_vecs > 0, sigma=sigma)
        # emphasize boundary points
        cmap /= cmap.max()
        # emphasize object interior points
        cmap[im.squeeze() > 0] += 0.5
        cmap += 0.5
        # very emphasize object contact points
        cmap += self._get_object_contacts(im.numpy())
        return torch.from_numpy(cmap).unsqueeze(0)

    def __call__(self, image_dict):
        for key in self.label_keys:
            if key not in image_dict:
                if not self.allow_missing_keys:
                    raise KeyError(
                        f"Key {key} not found in data. Available keys are {image_dict.keys()}"
                    )
                continue
            im = image_dict.pop(key)
            im = im.as_tensor() if isinstance(im, MetaTensor) else im
            im_numpy = im.numpy().astype(int).squeeze()
            skel = self.shrink(im_numpy)
            skel_edt = torch.from_numpy(edt.edt(skel > 0)).unsqueeze(0)
            skel_edt[skel_edt == 0] = -10
            skel = torch.from_numpy(skel)
            embed = self.embed_from_skel(skel, im.squeeze(0).clone())
            cmap = self._get_cmap(skel_edt.squeeze(), im)
            bound = torch.from_numpy(find_boundaries(im_numpy)).unsqueeze(0)
            image_dict[key] = torch.cat([skel_edt, im > 0, embed, bound, cmap]).float()
        return image_dict


class InstanceSegRandFlipd(RandomizableTransform):
    """Flipping Augmentation for InstanceSeg training.

    When flipping ground truths generated by `InstanceSegPreprocessD`, the sign of gradients have
    to be changed after flipping.
    """

    def __init__(
        self,
        spatial_axis: int,
        label_keys: Union[str, Sequence[str]] = [],
        image_keys: Union[str, Sequence[str]] = [],
        prob: float = 0.5,
        dim: int = 3,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        --------------
        spatial_axis:int
            axis to flip across
        label_keys:Union[str, Sequence[str]]=[]
            key or list of keys generated by InstanceSegPreprocessD to flip
        image_keys:Union[str, Sequence[str]]=[]
            key or list of keys NOT generated by InstanceSegPreprocessd to flip
        prob:float=0.1
            probability of flipping
        dim:int=3
            spatial dimensions of images
        allow_missing_keys:bool=False
            Whether to raise error if a provided key is missing
        """
        super().__init__()
        self.image_keys = (
            image_keys if isinstance(image_keys, (list, ListConfig)) else [image_keys]
        )
        self.label_keys = (
            label_keys if isinstance(label_keys, (list, ListConfig)) else [label_keys]
        )
        self.dim = dim
        self.allow_missing_keys = allow_missing_keys
        self.flipper = Flip(spatial_axis)
        self.prob = prob
        self.spatial_axis = spatial_axis

    def _flip(self, img, is_label):
        img = self.flipper(img)
        if is_label:
            assert (
                img.shape[0] == 4 + self.dim
            ), f"Expected generated InstanceSeg ground truth to have {4+self.dim} channels, got {img.shape[0]}"
            flipped_flows = img[2 : 2 + self.dim]
            flipped_flows[self.spatial_axis] *= -1
            img[2 : 2 + self.dim] = flipped_flows
        return img

    def __call__(self, image_dict):
        do_flip = self.R.rand() < self.prob
        if do_flip:
            for key in self.label_keys + self.image_keys:
                if key in image_dict:
                    image_dict[key] = self._flip(image_dict[key], key in self.label_keys)
                elif not self.allow_missing_keys:
                    raise KeyError(
                        f"Key {key} not found in data. Available keys are {image_dict.keys()}"
                    )
        return image_dict


class InstanceSegLoss:
    """Loss function for InstanceSeg."""

    def __init__(self, dim: int = 3):
        """
        Parameters
        --------------
        dim:int=3
            Spatial dimension of input images.
        """
        self.dim = dim
        self.skeleton_loss = CMAP_loss(torch.nn.MSELoss(reduction="none"))
        self.vector_loss = CMAP_loss(torch.nn.MSELoss(reduction="none"))
        self.boundary_loss = CMAP_loss(torch.nn.BCEWithLogitsLoss(reduction="none"))
        self.semantic_loss = TverskyLoss(sigmoid=True)

    def __call__(self, y_hat, y):
        """
        Parameters
        --------------
        y: ND-array, float
            y[:,0] skeleton
            y[:,1] semantic_segmentation
            y[:,2:2+self.dim] embedding
            y[:, -2] boundary segmentation
            y[:, -1] costmap for vector loss

        y_hat: ND-array, float
            y[:,0] skeleton
            y[:,1] semantic_segmentation
            y[:,2:2+self.dim embedding
            y[:, -1] boundary

        """
        cmap = y[:, -1:]
        skeleton_loss = self.skeleton_loss(y_hat[:, :1], y[:, :1], cmap)
        semantic_loss = self.semantic_loss(y_hat[:, 1:2], y[:, 1:2])
        boundary_loss = self.boundary_loss(y_hat[:, -1:], y[:, -2:-1], cmap)
        vector_loss = self.vector_loss(y_hat[:, 2:-1], y[:, 2:-2], cmap) * 10
        return vector_loss + skeleton_loss + semantic_loss + boundary_loss


class InstanceSegCluster:
    """
    Clustering for InstanceSeg - finds skeletons and assigns semantic points to skeleton based on spatial embedding and nearest neighbor distances.
    """

    def __init__(
        self,
        dim: int = 3,
        anisotropy: float = 2.6,
        skel_threshold: float = 0,
        semantic_threshold: float = 0,
        min_size: int = 1000,
        distance_threshold: int = 100,
    ):
        self.dim = dim
        self.anisotropy = torch.as_tensor([anisotropy if dim == 3 else 1] + [1] * (dim - 1))
        self.skel_threshold = skel_threshold
        self.semantic_threshold = semantic_threshold
        self.min_size = min_size
        self.distance_threshold = distance_threshold

    def _get_point_embeddings(self, object_points, skeleton_points):
        """
        object_points: (N, dim) array of embedded points from semantic segmentation
        skeleton_points: (N, dim) array of points on skeleton boundary
        """
        tree = KDTree(skeleton_points)
        dist, idx = tree.query(object_points)
        return dist, tree.data[idx].T.astype(int)

    def kd_clustering(self, embeddings, skel):
        """assign embedded points to closest skeleton."""
        skel = find_boundaries(skel, mode="inner") * skel  # propagate labels to boundaries
        skel_points = np.stack(skel.nonzero()).T
        embed_points = np.stack(embeddings).T
        (
            dist_to_closest_skel,
            closest_skel_point_to_embedding,
        ) = self._get_point_embeddings(embed_points, skel_points)
        embedding_labels = skel[tuple(closest_skel_point_to_embedding[:3])]
        # remove points too far from any skeleton
        embedding_labels[dist_to_closest_skel > self.distance_threshold] = 0
        return embedding_labels

    def _get_largest_cc(self, im):
        im = label(im)
        largest_cc = np.argmax(np.bincount(im.flatten())[1:]) + 1
        return im == largest_cc

    def __call__(self, image):
        image = image.detach().cpu().float()
        skel = image[0].numpy()
        semantic = image[1]
        embedding = image[2 : 2 + self.dim]
        # z embeddings are anisotropic, have to adjust to coordinates in real space, not pixel space
        anisotropic_shape = torch.as_tensor(semantic.shape).mul(self.anisotropy)

        coordinates = torch.stack(
            torch.meshgrid(
                *[
                    torch.linspace(0, anisotropic_shape[i] - 1, semantic.shape[i])
                    for i in range(self.dim)
                ]
            )
        )
        embedding += coordinates

        # create instances from skeletons, removing small, anomalous skeletons
        skel = apply_hysteresis_threshold(
            skel, high=self.skel_threshold, low=self.skel_threshold - 1
        )
        skel = label(skel)
        skel = remove_small_objects(skel, self.min_size)

        semantic = semantic > self.semantic_threshold
        # if only one skeleton, return largest connected component of semantic segmentation
        if len(np.unique(skel)) == 2:
            return self._get_largest_cc(semantic).astype(np.uint8)

        out = np.zeros_like(semantic, dtype=np.uint16)
        # find pixel coordinates pointed to by each z, y, x point within semantic segmentation
        embeddings = []
        for i in range(embedding.shape[0]):
            dim_embed = embedding[i][semantic] / self.anisotropy[i]
            dim_embed = dim_embed.clip(0, semantic.shape[i] - 1).round().int()
            embeddings.append(dim_embed)

        # assign each embedded point the label of the closest skeleton
        labeled_embed = self.kd_clustering(embeddings, skel)
        # propagate embedding label to semantic segmentation
        out[semantic] = labeled_embed
        return out
