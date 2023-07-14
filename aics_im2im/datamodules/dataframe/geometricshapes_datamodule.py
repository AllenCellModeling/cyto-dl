import numpy as np
import torch
import torch_geometric.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_geometric.datasets import GeometricShapes


class GeometricShapesDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        num_points: int,
        rotate: bool,
        jitter: bool,
        cond_class: int,
        couple_rot: bool,
        num_samples_per_class: int,
        eval_samples_per_class: int,
        num_classes: int,
        path="/allen/aics/modeling/ritvik/projects/PointGPT/data/GeometricShapes-34/geometricshapes_pc/",
        return_id=False,
        x_label=None,
        y_label=None,
        cond_label=None,
    ):
        """
        num_points: number of points sampled from PC
        rotate: whether to apply random rotations during training
        jitter: whether to apply random jitter during training
        cond_class: whether to generate a reference shape from a particular class
        couple_rot: whether to couple rotations of reference shape and actual shape
        num_samples_per_class: how many samples per class durng training
        eval_samples_per_clas: how many samples per class during eval
        num_classes: how many classes of shapes
        labels: key labels for all inputs
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_samples_per_class = eval_samples_per_class
        self.loader_fnc = DataLoader
        self.num_points = num_points
        self.path = path
        self.rotate = rotate
        self.jitter = jitter
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.cond_class = cond_class
        self.couple_rot = couple_rot
        self.return_id = return_id

        self.x_label = x_label
        self.y_label = y_label
        self.cond_label = cond_label

    def train_dataloader(self):
        self.train_dataset = RotGeometricShapes(
            self.num_points,
            False,
            False,
            self.num_samples_per_class,
            self.num_classes,
            self.cond_class,
            self.couple_rot,
            self.return_id,
            self.path,
            True,
        )
        dataloader = self.loader_fnc(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self):
        self.val_dataset = RotGeometricShapes(
            self.num_points,
            False,
            False,
            self.eval_samples_per_class,
            self.num_classes,
            self.cond_class,
            self.couple_rot,
            self.return_id,
            self.path,
            False,
        )
        dataloader = self.loader_fnc(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        self.test_dataset = RotGeometricShapes(
            self.num_points,
            False,
            False,
            self.eval_samples_per_class,
            self.num_classes,
            self.cond_class,
            self.couple_rot,
            self.return_id,
            self.path,
            False,
        )
        dataloader = self.loader_fnc(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    # def test_dataloader(self):
    #     self.test_dataset = RotGeometricShapes(
    #         self.num_points, False, False, self.eval_samples_per_class,
    #         self.num_classes, self.cond_class, self.couple_rot, self.x_label, self.y_label, self.cond_label
    #     )
    #     dataloader = self.loader_fnc(
    #         dataset=self.test_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #         shuffle=False,
    #     )
    #     return dataloader


# class RotGeometricShapes(Dataset):
#     def __init__(self, num_points, rotate, jitter, num_samples_per_class, num_classes, cond_class=None, couple_rot=False, x_label=None, y_label=None, cond_label=None):
#         self.num_points = num_points
#         self.rotate = rotate
#         self.jitter = jitter
#         self.num_samples_per_class = num_samples_per_class
#         self.num_classes = num_classes
#         self.cond_class = cond_class
#         self.couple_rot = couple_rot
#         # self.transform = T.Compose([T.SamplePoints(num=self.num_points), T.RandomRotate(self.rotate_range)])
#         self.transform = T.Compose([T.SamplePoints(num=self.num_points)])
#         self.x_label = x_label
#         self.y_label = y_label
#         self.cond_label = cond_label
#         self.data = []
#         self.label = []
#         dataset = GeometricShapes(root='/tmp/GeomShapes', transform=self.transform)
#         for i in range(self.num_classes):
#             for j in range(self.num_samples_per_class):
#                 self.data.append(dataset[i].pos)
#                 self.label.append(dataset[i].y)

#         if self.cond_class is not None:
#             self.ref = []
#             for i in range(self.num_classes * self.num_samples_per_class):
#                 self.ref.append(dataset[self.cond_class].pos)


#     def __len__(self):
#         return self.num_classes * self.num_samples_per_class

#     def __getitem__(self, item):
#         x = self.data[item].float()
#         y = self.label[item].float()
#         if self.rotate is not None:
#             x, rotation_matrix, theta = rotate_pointcloud(x.numpy(), None, True)
#             x = torch.from_numpy(x)
#         else:
#             theta = 0
#         if self.jitter:
#             x = torch.from_numpy(jitter_pointcloud(x.numpy()))

#         if self.cond_class:
#             c = self.ref[item].float()
#             if self.couple_rot:
#                 c = rotate_pointcloud(c.numpy(), rotation_matrix, False)
#                 theta_c = theta
#             else:
#                 c, rotation_matrix_c, theta_c= rotate_pointcloud(c.numpy(), None, True)
#             c = torch.from_numpy(c)
#             return {self.x_label: x, self.y_label: y, self.cond_label: c, 'rotation': theta,  'rotation_c': theta_c }
#         else:
#             return {self.x_label: x, self.y_label: y, 'rotation': theta}


class RotGeometricShapes(Dataset):
    def __init__(
        self,
        num_points,
        rotate,
        jitter,
        num_samples_per_class,
        num_classes,
        cond_class=None,
        couple_rot=False,
        return_id=False,
        path=None,
        train=True,
        conditional=False,
    ):
        self.path = path
        self.num_points = num_points
        self.rotate = rotate
        self.jitter = jitter
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.cond_class = cond_class
        self.couple_rot = couple_rot
        self.conditional = conditional
        # self.transform = T.Compose([T.SamplePoints(num=self.num_points), T.RandomRotate(self.rotate_range)])
        self.transform = T.Compose([T.SamplePoints(num=self.num_points)])

        self.return_id = return_id

        if self.path is not None:
            self.data = []
            # self.items = os.listdir(self.path)
            if train:
                self.items = [str(i) + ".npy" for i in range(1200)]
                scale = 200
            else:
                self.items = [str(i + 1300) + ".npy" for i in range(120)]
                scale = 20
            self.len = len(self.items)
            self.label = []
            for i in range(len(self.items)):
                self.data.append(torch.from_numpy(np.load(self.path + self.items[i])))
                self.label.append(torch.tensor(int(i / scale)))

            if self.conditional:
                pass
        else:
            self.len = self.num_classes * self.num_samples_per_class
            self.data = []
            self.label = []
            dataset = GeometricShapes(root="/tmp/GeomShapes", transform=self.transform)
            for i in range(self.num_classes):
                this_i = dataset[i].pos * 10
                for j in range(self.num_samples_per_class):
                    self.data.append(this_i)
                    self.label.append(dataset[i].y)

            if self.cond_class is not None:
                self.ref = []
                for i in range(self.num_classes * self.num_samples_per_class):
                    self.ref.append(dataset[self.cond_class].pos)

    def __len__(self):
        return self.len

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, item):
        x = self.data[item].float().numpy()
        x = self.pc_norm(x) * 2
        x = torch.from_numpy(x).float()
        y = self.label[item].float()
        # import ipdb
        # ipdb.set_trace()
        if self.rotate is not False:
            x, rotation_matrix, theta = rotate_pointcloud(x.numpy(), None, True)
            x = torch.from_numpy(x)
        else:
            theta = 0
        if self.jitter:
            x = torch.from_numpy(jitter_pointcloud(x.numpy()))

        if self.cond_class:
            c = self.ref[item].float()
            if self.couple_rot:
                c = rotate_pointcloud(c.numpy(), rotation_matrix, False)
                theta_c = theta
            else:
                c, rotation_matrix_c, theta_c = rotate_pointcloud(c.numpy(), None, True)
            c = torch.from_numpy(c)
            if self.return_id:
                return {
                    "pcloud": x,
                    "cond": c,
                    "class": y,
                    "rotation": theta,
                    "rotation_c": theta_c,
                }
            else:
                return {"pcloud": x, "cond": c}
        else:
            if self.return_id:
                return {"pcloud": x, "class": y, "rotation": theta}
            else:
                return {"pcloud": x}


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
    # import ipdb
    # ipdb.set_trace()
    if return_rot:
        return pointcloud_rotated, rotation_matrix, theta

    return pointcloud_rotated
