from typing import Dict, List, Optional, Sequence, Union

import edt
import numpy as np
import torch

# implements a nan-ignoring convolution
from astropy.convolution import convolve
from monai.data import MetaTensor
from monai.losses import TverskyLoss
from monai.transforms import Flip, RandomizableTransform, Transform
from omegaconf import ListConfig
from scipy.ndimage import binary_dilation, binary_erosion, find_objects, label
from scipy.spatial import KDTree
from skimage.filters import gaussian
from skimage.morphology import ball, disk, remove_small_objects, skeletonize
from skimage.segmentation import find_boundaries, relabel_sequential
from tqdm import tqdm

from cyto_dl.nn.losses.loss_wrapper import CMAP_loss


def pad_slice(s, padding, constraints):
    # pad slice by padding subject to image size constraints
    new_slice = []
    for slice_part, c in zip(s, constraints):
        start = max(0, slice_part.start - padding)
        stop = min(c, slice_part.stop + padding)
        new_slice.append(slice(start, stop, None))
    return tuple(new_slice)


class InstanceSegPreprocessd(Transform):
    def __init__(
        self,
        label_keys: Union[Sequence[str], str],
        kernel_size: int = 3,
        thin: int = 5,
        dim: int = 3,
        anisotropy: float = 2.6,
        keep_largest: bool = True,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        label_keys: Union[Sequence[str], str]
            Keys of instance segmentations in input dictionary to convert to Instance Seg ground truth images.
        kernel_size: int=3
            Size of kernel for gaussian smoothing of flows
        thin: int=5
            Amount to thin to create psuedo-skeleton
        dim:int=3
            Spatial dimension of images
        anisotropy:float=2.6
            Anisotropy of images
        keep_largest:bool=True
            Whether to keep only the largest connected component of each label
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
        self.anisotropy = np.array([anisotropy, 1, 1]) if dim == 3 else np.array([1, 1])
        self.thin = thin
        self.keep_largest = keep_largest

    def shrink(self, im):
        """Topology-preserving thinning of a binary image."""
        skel = np.zeros_like(im)
        regions = find_objects(im)
        for lab, coords in enumerate(regions, start=1):
            if coords is None:
                continue
            # add 1 pix boundary to prevent "ball + stick" artifacts
            coords = pad_slice(coords, 2, im.shape)
            skel[coords] += self.topology_preserving_thinning(im[coords] == lab)
        return skel * im  # relabel shrunk object

    def skeleton_tall(self, img, max_label):
        """Skeletonize 3d image with increased thickness in z."""
        if max_label == 0 or self.dim == 2:
            return skeletonize(img)
        tall_skeleton = np.stack([skeletonize(np.max(img, 0))] * img.shape[0])
        return tall_skeleton

    def label_2d(self, img):
        """Dim = 2: return labeled image dim = 3: label each z slice separately."""
        if self.dim == 2:
            out, _ = label(img)
            return out
        out = np.zeros_like(img, dtype=np.int16)
        for z in range(img.shape[0]):
            lab, _ = label(img[z])
            lab[lab > 0] += np.max(out)
            out[z] = lab
        return out

    def topology_preserving_thinning(self, bw, min_size=100):
        """Topology-preserving thinning of a binary image.

        Use skeleton to bridge gaps created by erosion.
        """
        # NOTE - keeping every self.thin slices does not maintain self.anisotropy ( keeping every self.anisotropy slices would). In practice, keeping every self.anisotropy slices is slower and favors z-gradients over xy.
        selem = ball(self.thin)[:: self.thin] if self.dim == 3 else disk(self.thin)
        eroded = binary_erosion(bw, selem, border_value=1)
        # only want to preserve connections between significantly-sized objects

        eroded = remove_small_objects(eroded, min_size)
        eroded, max_label = label(eroded)
        # single object is preserved by erosion
        if max_label == 1:
            return eroded

        skel = self.skeleton_tall(bw, max_label)

        if max_label == 0:
            return skel
        # if erosion separates object into multiple pieces, use skeleton to bridge those pieces into single object
        # 1. isolate pieces of skeleton that are outside of eroded objects (i.e. could bridge between objects)
        skel[eroded != 0] = 0
        skel = self.label_2d(skel)

        for i in np.unique(skel)[1:]:
            # 3. find number of non-background objects overlapped by piece of skeleton, add back in pieces that overlap multiple obj
            dilation_selem = np.expand_dims(disk(3), 0)
            dilated_skel = binary_dilation(skel == i, dilation_selem)
            n_obj_masked = np.sum(np.unique(eroded[dilated_skel]) > 0)
            if n_obj_masked > 1:
                eroded += dilated_skel
        # make sure dilated skeleton is within object bounds by 1 pix so vectors can point to it
        one_erode = binary_erosion(bw, border_value=1)
        eroded[one_erode == 0] = 0
        return eroded > 0

    def _get_point_embeddings(self, object_points, skeleton_points):
        """Finds closest skeleton point to each object point using KDTree."""
        tree = KDTree(skeleton_points)
        dist, idx = tree.query(object_points)
        return torch.from_numpy(tree.data[idx]).T.float()

    def smooth_embedding(self, embedding):
        """Smooths embedding by convolving with a mean kernel, excluding non-object pixels."""
        kernel = np.ones([self.kernel_size] * self.dim) / (self.kernel_size**self.dim)
        embedding[embedding == 0] = np.nan
        for i in range(embedding.shape[0]):
            conv_embed = convolve(embedding[i], kernel, boundary="extend")
            conv_embed[np.isnan(embedding[i])] = 0
            embedding[i] = conv_embed
        return embedding

    def embed_from_skel(self, skel: np.ndarray, iseg: np.ndarray):
        """Find per-pixel embedding vector to closest point on skeleton."""
        iseg[skel != 0] = 0
        # 3ZYX vector field for 3d, 2YX for 2d
        embed = torch.zeros([self.dim] + [iseg.shape[i] for i in range(self.dim)])
        skel_boundary = find_boundaries(skel, mode="inner") * skel  # propagate labels

        regions = find_objects(iseg)
        for lab, coords in enumerate(regions, start=1):
            if coords is None:
                continue
            seg_crop = iseg[coords]
            # find objects + np.where is much faster than just np.where on full fov
            object_points = np.asarray(np.where(seg_crop == lab))
            skel_points = np.asarray(np.where(skel_boundary[coords] == lab))
            if skel_points[0].size == 0:
                continue
            # distances should take into account z anisotropy and be in n_points x n_dims array
            point_embeddings = self._get_point_embeddings(
                object_points.T * self.anisotropy, skel_points.T * self.anisotropy
            )

            # smooth embeddings per-object to avoid smearing of boundaries across objects
            crop_embedding = np.zeros((self.dim, *seg_crop.shape))

            if len(object_points) == 2:
                crop_embedding[:, object_points[0], object_points[1]] = point_embeddings
            elif len(object_points) == 3:
                crop_embedding[:, object_points[0], object_points[1], object_points[2]] = (
                    point_embeddings
                )

            crop_embedding = torch.from_numpy(self.smooth_embedding(crop_embedding))

            # turn spatial embedding into offset vector by subtracting pixel coordinates
            anisotropic_shape = torch.as_tensor(seg_crop.shape).mul(
                torch.from_numpy(self.anisotropy)
            )
            coordinates = torch.stack(
                torch.meshgrid(
                    *[
                        torch.linspace(0, anisotropic_shape[i] - 1, seg_crop.shape[i])
                        for i in range(self.dim)
                    ]
                )
            )
            crop_embedding[crop_embedding != 0] -= coordinates[crop_embedding != 0]

            # pad coords with channel dimension
            embed[(slice(None),) + coords] += crop_embedding
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
        points_with_vecs = im.copy()
        points_with_vecs[skel_edt > 0] = 0
        add_in_thin = np.logical_and(skel_edt > 0, skel_edt < 3)
        points_with_vecs = np.logical_or(points_with_vecs, add_in_thin)
        sigma = np.asarray([2] * self.dim) / self.anisotropy
        sigma = np.maximum(sigma, np.ones(self.dim))
        cmap = gaussian(points_with_vecs > 0, sigma=sigma)
        # emphasize boundary points
        cmap /= cmap.max()
        # emphasize object interior points
        cmap[im.squeeze() > 0] += 0.5
        cmap += 0.5
        # very emphasize object contact points
        cmap += self._get_object_contacts(im)
        return torch.from_numpy(cmap).unsqueeze(0)

    def keep_largest_cc(self, img):
        regions = find_objects(img)
        new_im = np.zeros_like(img, dtype=img.dtype)
        for lab, coords in enumerate(regions, start=1):
            if lab == 0:
                continue
            labeled_crop, n_labels = label(img[coords] == lab)
            if n_labels > 1:
                largest_cc = np.argmax(np.bincount(labeled_crop.flat)[1:]) + 1
                largest_cc = (labeled_crop == largest_cc) * lab
            else:
                largest_cc = (img[coords] == lab) * lab
            new_im[coords] += largest_cc
        return new_im

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
            im = im.numpy().astype(int).squeeze()
            if self.keep_largest:
                im = self.keep_largest_cc(im)
            im, _, _ = relabel_sequential(im)
            skel = self.shrink(im)
            skel_edt = torch.from_numpy(edt.edt(skel > 0)).unsqueeze(0)
            skel_edt[skel_edt == 0] = -10
            embed = self.embed_from_skel(skel, im.copy())
            cmap = self._get_cmap(skel_edt.squeeze(), im)
            bound = torch.from_numpy(find_boundaries(im, mode="inner")).unsqueeze(0)
            semantic_seg = torch.from_numpy(im > 0).unsqueeze(0)
            image_dict[key] = torch.cat([skel_edt, semantic_seg, embed, bound, cmap]).float()
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

    def __init__(self, dim: int = 3, weights: Optional[Dict[str, float]] = {}):
        """
        Parameters
        --------------
        dim:int=3
            Spatial dimension of input images.
        weights:Optional[Dict[str, float]]={}
            Dictionary of weights for each loss component.
        """
        self.dim = dim
        self.skeleton_loss = CMAP_loss(torch.nn.MSELoss(reduction="none"))
        self.vector_loss = CMAP_loss(torch.nn.MSELoss(reduction="none"))
        self.boundary_loss = CMAP_loss(torch.nn.BCEWithLogitsLoss(reduction="none"))
        self.semantic_loss = TverskyLoss(sigmoid=True)
        self.weights = weights

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
        skeleton_loss = self.skeleton_loss(y_hat[:, :1], y[:, :1], cmap) * float(
            self.weights.get("skeleton", 1.0)
        )
        semantic_loss = self.semantic_loss(y_hat[:, 1:2], y[:, 1:2]) * float(
            self.weights.get("semantic", 40.0)
        )
        boundary_loss = self.boundary_loss(y_hat[:, -1:], y[:, -2:-1], cmap) * float(
            self.weights.get("boundary", 1.0)
        )
        vector_loss = self.vector_loss(y_hat[:, 2:-1], y[:, 2:-2], cmap) * float(
            self.weights.get("vector", 10.0)
        )
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
        progress: bool = True,
    ):
        self.dim = dim
        self.anisotropy = np.array([anisotropy if dim == 3 else 1] + [1] * (dim - 1))
        self.skel_threshold = skel_threshold
        self.semantic_threshold = semantic_threshold
        self.min_size = min_size
        self.distance_threshold = distance_threshold
        self.progress = progress

    def _get_point_embeddings(self, object_points, skeleton_points):
        """
        object_points: (N, dim) array of embedded points from semantic segmentation
        skeleton_points: (N, dim) array of points on skeleton boundary
        """
        tree = KDTree(skeleton_points)
        dist, idx = tree.query(object_points)
        return dist, tree.data[idx].T.astype(int)

    def kd_clustering(self, embeddings, skel):
        """Assign embedded points to closest skeleton."""
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

    def remove_small_skeletons(self, skel):
        """Remove small skeletons below self.min_size that are not touching the edge of the
        image."""
        skel_removed = skel.copy()
        regions = find_objects(skel)
        for lab, coords in enumerate(regions, start=1):
            if coords is None:
                continue
            is_edge = np.any(
                [np.logical_or(s.start == 0, s.stop >= c) for s, c in zip(coords, skel.shape)]
            )
            if not is_edge and np.sum(skel[coords]) < self.min_size:
                skel_removed[coords][skel[coords] == lab] = 0

        return skel_removed

    def cluster_object(self, semantic, skel, embedding):
        skel[semantic == 0] = -np.inf
        # create instances from skeletons, removing small, anomalous skeletons
        skel, _ = label(skel > self.skel_threshold)
        skel = self.remove_small_skeletons(skel)
        num_objects = len(np.unique(skel)) - 1

        if num_objects == 0:
            # don't include objects corresponding to bad skeletons in final segmentation
            return np.zeros_like(semantic)
        elif num_objects == 1:
            # if only one skeleton, return largest connected component of semantic segmentation
            return semantic

        # z embeddings are anisotropic, have to adjust to coordinates in real space, not pixel space
        anisotropic_shape = np.array(semantic.shape) * self.anisotropy
        coordinates = np.stack(
            np.meshgrid(
                *[
                    np.linspace(0, anisotropic_shape[i] - 1, semantic.shape[i])
                    for i in range(self.dim)
                ],
                indexing="ij",
            )
        )
        embedding += coordinates
        semantic = np.logical_and(semantic, skel == 0)

        # find pixel coordinates pointed to by each z, y, x point within semantic segmentation and outside skeleton
        embeddings = []
        for i in range(embedding.shape[0]):
            dim_embed = embedding[i][semantic] / self.anisotropy[i]
            dim_embed = np.clip(dim_embed, 0, semantic.shape[i] - 1).round().astype(int)
            embeddings.append(dim_embed)

        # assign each embedded point the label of the closest skeleton
        labeled_embed = self.kd_clustering(embeddings, skel)
        # propagate embedding label to semantic segmentation
        skel[semantic] = labeled_embed
        out, _, _ = relabel_sequential(skel)
        return out

    def __call__(self, image):
        image = image.detach().half()
        naive_labeling, _ = label((image[1] > self.semantic_threshold).cpu())
        skel = image[0].cpu().numpy()
        embedding = image[2 : 2 + self.dim].cpu().numpy()

        regions = enumerate(find_objects(naive_labeling), start=1)

        highest_cell_idx = 0
        out_image = np.zeros_like(naive_labeling, dtype=np.uint16)
        for val, region in tqdm(regions) if self.progress else regions:
            region = pad_slice(region, 1, naive_labeling.shape)
            mask = self.cluster_object(
                (naive_labeling[region] == val).copy(),
                skel[region].copy(),
                embedding[(slice(None),) + region].copy(),
            )
            mask = mask.astype(np.uint16)
            max_mask = np.max(mask)
            mask[mask > 0] += highest_cell_idx
            out_image[region] += mask
            highest_cell_idx += max_mask
        return out_image
