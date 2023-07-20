from .transforms import (
    SubsamplePoints,
    PointcloudNoise,
    SubsamplePointcloud,
)
from .fields import (
    PointsField,
    PointCloudField,
    PartialPointCloudField,
    IndexField,
)
from torchvision import transforms


def get_data_fields(
    mode, points_subsample, input_type, points_file, multi_files, points_iou_file
):
    """Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    """
    points_transform = SubsamplePoints(points_subsample)

    input_type = input_type
    fields = {}
    if points_file is not None:
        fields["points"] = PointsField(
            points_file,
            points_transform,
            unpackbits=False,
            multi_files=multi_files,
        )

    if mode in ("val", "test"):
        if points_iou_file is not None:
            fields["points_iou"] = PointsField(
                points_iou_file,
                unpackbits=False,
                multi_files=multi_files,
            )

    return fields


def get_inputs_field(
    mode,
    input_type,
    pointcloud_n,
    pointcloud_noise,
    pointcloud_file,
    multi_files,
    part_ratio,
    partial_type,
):
    """Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    """
    if input_type is None:
        inputs_field = None
    elif input_type == "pointcloud":
        transform = transforms.Compose(
            [SubsamplePointcloud(pointcloud_n), PointcloudNoise(pointcloud_noise)]
        )
        inputs_field = PointCloudField(
            pointcloud_file, transform, multi_files=multi_files
        )
    elif input_type == "partial_pointcloud":
        transform = transforms.Compose(
            [
                SubsamplePointcloud(pointcloud_n),
                PointcloudNoise(pointcloud_noise),
            ]
        )
        inputs_field = PartialPointCloudField(
            pointcloud_file,
            transform,
            multi_files=multi_files,
            part_ratio=part_ratio,
            partial_type=partial_type,
        )
    elif input_type == "idx":
        inputs_field = IndexField()
    else:
        raise ValueError("Invalid input type (%s)" % input_type)
    return inputs_field


def normalize_3d_coordinate(p, padding=0.1):
    """Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    p_nor = p / (1 + padding + 10e-4)  # (-0.5, 0.5)
    p_nor = p_nor + 0.5  # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def normalize_coordinate(p, padding=0.1, plane="xz"):
    """Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    """
    if plane == "xz":
        xy = p[:, :, [0, 2]]
    elif plane == "xy":
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
    xy_new = xy_new + 0.5  # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def coordinate2index(x, reso, coord_type="2d"):
    """Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    """
    x = (x * reso).long()
    if coord_type == "2d":  # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == "3d":  # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


def coord2index(p, vol_range, reso=None, plane="xz"):
    """Normalize coordinate to [0, 1] for sliding-window experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): points
        vol_range (numpy array): volume boundary
        reso (int): defined resolution
        plane (str): feature type, ['xz', 'xy', 'yz'] - canonical planes; ['grid'] - grid volume
    """
    # normalize to [0, 1]
    x = normalize_coord(p, vol_range, plane=plane)

    if isinstance(x, np.ndarray):
        x = np.floor(x * reso).astype(int)
    else:  # * pytorch tensor
        x = (x * reso).long()

    if x.shape[1] == 2:
        index = x[:, 0] + reso * x[:, 1]
        index[index > reso**2] = reso**2
    elif x.shape[1] == 3:
        index = x[:, 0] + reso * (x[:, 1] + reso * x[:, 2])
        index[index > reso**3] = reso**3

    return index[None]


def update_reso(reso, depth):
    """Update the defined resolution so that UNet can process.

    Args:
        reso (int): defined resolution
        depth (int): U-Net number of layers
    """
    base = 2 ** (int(depth) - 1)
    if ~(reso / base).is_integer():  # when this is not integer, U-Net dimension error
        for i in range(base):
            if ((reso + i) / base).is_integer():
                reso = reso + i
                break
    return reso
