import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import json
import os
from pyntcloud import PyntCloud
import point_cloud_utils as pcu
import pandas as pd
from typing import Optional


class CellPackDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        num_points: int,
        num_points_ref: int,
        packing_rotations: list = [0, 1, 2],
        packing_rules: list = [
            "planar_gradient_0deg",
            "planar_gradient_45deg",
            "planar_gradient_90deg",
        ],
        structure_path="/allen/aics/animated-cell/Saurabh/forRitvik/pcna_cellPACK/out/pcna/",
        ref_path="/allen/aics/modeling/ritvik/forSaurabh/",
        return_id=False,
        x_label=None,
        ref_label=None,
        rotation_augmentations: Optional[int] = None,
        jitter_augmentations: Optional[int] = None,
    ):
        """ """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loader_fnc = DataLoader
        self.num_points = num_points
        self.return_id = return_id
        self.rotation_augmentations = rotation_augmentations
        self.jitter_augmentations = jitter_augmentations
        self.num_points_ref = num_points_ref
        self.packing_rotations = packing_rotations
        self.packing_rules = packing_rules
        self.x_label = x_label
        self.ref_label = ref_label
        self.structure_path = structure_path
        self.ref_path = ref_path

    def _get_dataset(self, split):
        return CellPackDataset(
            self.num_points,
            self.num_points_ref,
            self.return_id,
            self.packing_rotations,
            self.packing_rules,
            self.structure_path,
            self.ref_path,
            split,
            self.x_label,
            self.ref_label,
            self.rotation_augmentations,
            self.jitter_augmentations,
        )

    def train_dataloader(self):
        self.train_dataset = self._get_dataset("train")
        dataloader = self.loader_fnc(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self):
        self.val_dataset = self._get_dataset("valid")
        dataloader = self.loader_fnc(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        self.test_dataset = self._get_dataset("test")
        dataloader = self.loader_fnc(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader


class CellPackDataset(Dataset):
    def __init__(
        self,
        num_points,
        num_points_ref,
        return_id=False,
        packing_rotations: list = [0, 1, 2],
        packing_rules: list = [
            "planar_gradient_0deg",
            "planar_gradient_45deg",
            "planar_gradient_90deg",
        ],
        structure_path="/allen/aics/animated-cell/Saurabh/forRitvik/pcna_cellPACK/out/pcna/",
        ref_path="/allen/aics/modeling/ritvik/forSaurabh/",
        split="train",
        x_label: str = "pcloud",
        ref_label: str = "nuc",
        rotation_augmentations: Optional[int] = None,
        jitter_augmentations: Optional[int] = None,
    ):
        self.x_label = x_label
        self.ref_label = ref_label
        self.num_points = num_points
        self.packing_rotations = packing_rotations
        self.packing_rules = packing_rules
        self.structure_path = structure_path
        self.ref_path = ref_path
        self.rotation_augmentations = rotation_augmentations
        self.jitter_augmentations = jitter_augmentations
        self.num_points_ref = num_points_ref

        self.return_id = return_id
        self.split = split
        items = os.listdir(self.structure_path)
        items = [i for i in items if "positions" in i]

        self._all_ids = list(
            set([i.split(".")[0].split("_")[-2].split("deg")[-1] for i in items])
        )

        _splits = {
            "train": self._all_ids[: int(0.7 * len(self._all_ids))],
            "valid": self._all_ids[
                int(0.7 * len(self._all_ids)) : int(0.85 * len(self._all_ids))
            ],
            "test": self._all_ids[int(0.85 * len(self._all_ids)) :],
        }

        self.ids = _splits[split]

        self.data = []
        self.ref = []
        self.id_list = []
        self.rule = []
        self.pack_rot = []
        self.rot = []
        self.jitter = []

        for this_id in self.ids:
            for rule in packing_rules:
                for rot in packing_rotations:
                    this_path = (
                        "positions_pcna_analyze_"
                        + f"{rule}"
                        + f"{this_id}"
                        + f"_{rot}"
                        + ".json"
                    )
                    with open(structure_path + this_path, "r") as f:
                        tmp = json.load(f)
                        points = pd.DataFrame()
                        points["x"] = [i[0] for i in tmp["0"]["nucleus_interior_pcna"]]
                        points["y"] = [i[1] for i in tmp["0"]["nucleus_interior_pcna"]]
                        points["z"] = [i[2] for i in tmp["0"]["nucleus_interior_pcna"]]

                        nuc_path = ref_path + f"{this_id}_{rot}.obj"
                        my_point_cloud = PyntCloud.from_file(nuc_path)
                        points_dna = my_point_cloud.points

                        if points_dna.shape[0] > num_points_ref:
                            points_dna = self._subsample(points_dna, num_points_ref)

                        if points.shape[0] > num_points:
                            points = self._subsample(points, num_points)

                        self.data.append(points.values)
                        self.ref.append(points_dna.values)
                        self.id_list.append(self._all_ids.index(this_id))
                        self.rule.append(self.packing_rules.index(rule))
                        self.pack_rot.append(self.packing_rotations.index(rot))
                        self.rot.append(np.array(0))
                        self.jitter.append(np.array(0))

                        if self.rotation_augmentations:
                            for i in range(self.rotation_augmentations):
                                (
                                    points_dna_rot,
                                    rotation_matrix,
                                    theta,
                                ) = rotate_pointcloud(
                                    points_dna.values, return_rot=True
                                )
                                points_rot = rotate_pointcloud(
                                    points.values, rotation_matrix, return_rot=False
                                )

                                if self.jitter_augmentations:
                                    for j in range(self.jitter_augmentations):
                                        self.data.append(jitter_pointcloud(points_rot))
                                        self.ref.append(
                                            jitter_pointcloud(points_dna_rot)
                                        )
                                        self.id_list.append(
                                            self._all_ids.index(this_id)
                                        )
                                        self.rule.append(self.packing_rules.index(rule))
                                        self.pack_rot.append(
                                            self.packing_rotations.index(rot)
                                        )
                                        self.rot.append(np.array(i))
                                        self.jitter.append(np.array(j))

                                else:
                                    self.data.append(points_rot)

                                    self.id_list.append(self._all_ids.index(this_id))
                                    self.rule.append(self.packing_rules.index(rule))
                                    self.pack_rot.append(
                                        self.packing_rotations.index(rot)
                                    )
                                    self.rot.append(np.array(i))
                                    self.jitter.append(np.array(0))

        self.len = len(self.data)
        self.label = []

    def __len__(self):
        return self.len

    def _subsample(self, points_dna, num_points_ref):
        v = points_dna.values
        idx = pcu.downsample_point_cloud_poisson_disk(
            v, num_samples=num_points_ref + 50
        )
        v_sampled = v[idx]
        points_dna = pd.DataFrame()
        points_dna["x"] = v_sampled[:, 0]
        points_dna["y"] = v_sampled[:, 1]
        points_dna["z"] = v_sampled[:, 2]
        points_dna = points_dna.sample(n=num_points_ref)
        return points_dna

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, item):
        x = self.data[item]
        # x = self.pc_norm(x)
        x = torch.from_numpy(x).float()
        ref = self.ref[item]
        # ref = self.pc_norm(ref)
        ref = torch.from_numpy(ref).float()

        if self.return_id:
            return {
                self.x_label: x,
                self.ref_label: ref,
                "CellId": torch.tensor(self.id_list[item]),
                "rule": torch.tensor(self.rule[item]),
                "packing_rotation": torch.tensor(self.pack_rot[item]),
                "rotation_aug": torch.tensor(self.rot[item]),
                "jitter_aug": torch.tensor(self.jitter[item]),
            }
        else:
            return {self.x_label: x, self.ref_label: ref}


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud, rotation_matrix=None, return_rot=False):
    pointcloud_rotated = pointcloud.copy()
    if rotation_matrix is None:
        theta = np.pi * 2 * np.random.choice(24) / 24
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
    pointcloud_rotated[:, [0, 1]] = pointcloud_rotated[:, [0, 1]].dot(
        rotation_matrix
    )  # random rotation (x,z)
    if return_rot:
        return pointcloud_rotated, rotation_matrix, theta

    return pointcloud_rotated
