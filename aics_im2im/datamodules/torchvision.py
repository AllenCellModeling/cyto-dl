import inspect
from contextlib import suppress

import torchvision.datasets as vdata
import torchvision.transforms.functional as F
from monai.data import Dataset, PersistentDataset
from monai.transforms import Compose
from omegaconf import ListConfig


def __getitem__(self, idx: int):
    img, target = self.__class__.__bases__[-1].__getitem__(self, idx)
    return {"image": F.to_tensor(img), "target": target}


def _cast_init_arg(value):
    if isinstance(value, inspect.Parameter):
        return value._default
    return value


class TorchvisionDatasetMeta(type):
    def __call__(cls, *args, **kwargs):
        init_args = inspect.signature(cls.__init__).parameters.copy()
        init_args.pop("self")

        cache_dir = kwargs.pop("cache_dir", None)
        transform = kwargs.pop("transform", None)

        keys = tuple(init_args.keys())

        user_init_args = {keys[ix]: arg for ix, arg in enumerate(args)}
        user_init_args.update(kwargs)
        init_args.update(user_init_args)
        init_args = {k: _cast_init_arg(v) for k, v in init_args.items()}

        obj = type.__call__(cls, **init_args)

        if transform is not None:
            if isinstance(transform, (list, tuple, ListConfig)):
                transform = Compose(transform)

        if cache_dir is not None:
            obj = PersistentDataset(obj, cache_dir=cache_dir, transform=transform)
        else:
            obj = Dataset(obj, transform=transform)

        return obj


_members = inspect.getmembers(vdata)
names = []
for name, member in _members:
    if inspect.isclass(member):
        with suppress(TypeError):
            subclass = TorchvisionDatasetMeta(name, (member,), {"__getitem__": __getitem__})
            globals()[name] = subclass
            names.append(name)
            # the following torchvision datasets aren't compatible with this approach:
            # CREStereo
            # CarlaStereo
            # ETH3DStereo
            # FallingThingsStereo
            # FlyingChairs
            # FlyingThings3D
            # HD1K
            # InStereo2k
            # Kitti2012Stereo
            # Kitti2015Stereo
            # KittiFlow
            # Middlebury2014Stereo
            # SceneFlowStereo
            # Sintel
            # SintelStereo

__all__ = names
