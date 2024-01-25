import numpy as np
from monai.transforms import RandomizableTransform, Transform
import torch
import copy

from monai.transforms import Resize
from monai.transforms import Transform

class GenerateLabels(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, img_dict):
        n_timepoints = img_dict['img'].shape[0]
        formation_idx = int(img_dict['formation'] - img_dict['track_start'])
        breakdown_idx = int(img_dict['breakdown'] - img_dict['track_start'])

        # 0: normal, 1: mitotic
        tp_labels = np.zeros(n_timepoints)
        if 0 <=formation_idx < len(tp_labels):
            tp_labels[:formation_idx] = 1

        if 0 <= breakdown_idx < len(tp_labels):
            tp_labels[breakdown_idx+1:] = 1
        img_dict['label'] = tp_labels
        return img_dict

class TrackCrop(RandomizableTransform):
    """
    Transform to randomly crop track to first n or last n timepoints
    """
    def __init__(self, p_first: float = 0.15, p_last: float =0.1, percentage = [0.2, 0.7]):
        """
        Parameters
        ----------
        p_first: float
            Probability to crop first n timepoints
        n_timepoints: int
            Number of timepoints to crop
        """
        super().__init__()
        self.p_first = p_first
        self.p_last = p_last
        self.percentage = np.array(percentage)

    def __call__(self, img_dict):
        new_im_dict = copy.deepcopy(img_dict)
        new_im = img_dict['img']

        n = self.R.randint(*(self.percentage*new_im.shape[0]).astype(int), size = 1)[0]
        n = max(20, n)
        #crop to start of track
        if self.R.random() < self.p_first:
            new_im_dict['img'] = new_im[:n]
            new_im_dict['label'] = new_im_dict['label'][:n]
        # crop to end of track
        elif self.R.random() < self.p_last:
            new_im_dict['img'] = new_im[-n:]
            new_im_dict['label'] = new_im_dict['label'][-n:]
        return new_im_dict

class PerChannel(Transform):
    def __init__(self, keys, transform):
        super().__init__()
        self.transform = transform
        self.keys= keys

    def __call__(self, img_dict):
        new_im_dict = copy.deepcopy(img_dict)

        for key in self.keys:
            for i in range(new_im_dict[key].shape[0]):
                new_im_dict[key][i] = self.transform(new_im_dict[key][i])
        return new_im_dict


class CropResize(RandomizableTransform):
    def __init__(self, keys, max_shift=8):
        super().__init__()
        self.keys = keys
        self.max_shift= max_shift
    def __call__(self, img_dict):
        new_im_dict = copy.deepcopy(img_dict)

        for key in self.keys:
            resizer = Resize(new_im_dict[key].shape[-2:])
            resized_movie = []
            for im in new_im_dict[key]:
                shift = self.R.randint(0, self.max_shift, size=4)
                im = im[shift[0]:im.shape[0]-shift[1], shift[2]:im.shape[1]-shift[3]]
                im = resizer(im.unsqueeze(0)).squeeze(0)
                resized_movie.append(im)
            new_im_dict[key] = torch.stack(resized_movie)
        return new_im_dict