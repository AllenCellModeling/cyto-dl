import numpy as np
from monai.transforms import RandomizableTransform, Transform
from skimage.filters import gaussian
import torch
import copy

# class WindowCrop(RandomizableTransform):
#     def __init__(
#         self,
#         window_size: int= 3,
#         num_samples: int = 16,
#         reweight: bool = True
#     ):
#         """
#         Parameters
#         ----------

#         """
#         super().__init__()
#         self.window_size = window_size
#         self.reweight = reweight
#         self.num_samples = num_samples


#     def __call__(self, img_dict):
#         n_timepoints = img_dict['img'].shape[0]
#         formation_idx = int(img_dict['formation'] - img_dict['track_start'])
#         breakdown_idx = int(img_dict['breakdown'] - img_dict['track_start'])
#         #give equal probability to formation, breakdown,and normal timepoints
#         p = np.ones(n_timepoints)
#         if self.reweight:
#             if formation_idx >= 0 and formation_idx < n_timepoints:
#                 p[formation_idx] = 300 * n_timepoints
#             if breakdown_idx < n_timepoints and breakdown_idx >= 0:
#                 p[breakdown_idx] = 150 * n_timepoints
#         p = p[:-self.window_size]
#         # upweight timepoints next to formation/breakdown
#         p = gaussian(p)
#         p /= p.sum()

#         data =[]
#         if self.num_samples > 0:
#             window_start = self.R.choice(n_timepoints - self.window_size, size=self.num_samples , p = p)
#         else:
#             window_start = np.arange(len(p) - self.window_size)
#         window_end = window_start + self.window_size
         
#         for ws, we in zip(window_start, window_end):
#             window_time_indices= np.arange(ws, we) + img_dict['track_start']

#             #label if the middle timepoint is formation or breakdown
#             label = 0
#             if img_dict['formation']  == window_time_indices[self.window_size//2 + 1]:
#                 label = 1
#             elif img_dict['breakdown']  == window_time_indices[self.window_size//2 + 1]:
#                 label = 2
#             data.append({'img': img_dict['img'][ws:we] , 'label': label})

#         return data



# class WindowCrop(RandomizableTransform):
#     """
#     Classify window + specify index where transition occurs 
#     """
#     def __init__(
#         self,
#         window_size: int= 3,
#         num_samples: int = 16,
#         reweight: bool = True
#     ):
#         """
#         Parameters
#         ----------

#         """
#         super().__init__()
#         self.window_size = window_size
#         self.reweight = reweight
#         self.num_samples = num_samples


#     def __call__(self, img_dict):
#         n_timepoints = img_dict['img'].shape[0]
#         formation_idx = int(img_dict['formation'] - img_dict['track_start'])
#         breakdown_idx = int(img_dict['breakdown'] - img_dict['track_start'])
#         #give equal probability to formation, breakdown,and normal timepoints
#         p = np.ones(n_timepoints)
#         if self.reweight:
#             if formation_idx >= 0 and formation_idx < n_timepoints:
#                 p[formation_idx] =  n_timepoints
#             if breakdown_idx < n_timepoints and breakdown_idx >= 0:
#                 p[breakdown_idx] =   n_timepoints
#         p = p[:-self.window_size]
#         # upweight timepoints next to formation/breakdown
#         p = gaussian(p)
#         p /= p.sum()

#         data =[]
#         if self.num_samples > 0:
#             window_start = self.R.choice(n_timepoints - self.window_size, size=self.num_samples , p = p)
#         else:
#             window_start = np.arange(len(p) - self.window_size)
#         window_end = window_start + self.window_size
         
#         for ws, we in zip(window_start, window_end):
#             window_time_indices= np.arange(ws, we) + img_dict['track_start']

#             #label if transition occurs in window
#             label = 0
#             index = -2
#             if img_dict['formation']  in window_time_indices:
#                 label = 1
#                 index = np.where(window_time_indices == img_dict['formation'])[0][0]
#             elif img_dict['breakdown']  in window_time_indices:
#                 label = 2
#                 index = np.where(window_time_indices == img_dict['breakdown'])[0][0]
            
#             data.append({'img': img_dict['img'][ws:we] , 'label': label, 'index': index})

#         return data

from scipy.ndimage import gaussian_filter
from monai.transforms import Transform
class GenerateLabelsSmoothFourClass(Transform):
    def __init__(self):
        super().__init__()

    def create_smoothed_label(self, indices, track_length ,sigma =0.8):
        label = np.zeros(track_length)
        label[indices] = 1
        label = gaussian_filter(label, sigma=sigma)
        label /= label.max()
        if len(indices)>1:
            start_indices = indices[::2]
            stop_indices = indices[1::2]
            for start, stop in zip(start_indices, stop_indices):
                label[start:stop+1] = 1
        
        return label

    def __call__(self, img_dict):
        n_timepoints = img_dict['img'].shape[0]
        formation_idx = int(img_dict['formation'] - img_dict['track_start']) if img_dict['formation']>0 else -1
        breakdown_idx = int(img_dict['breakdown'] - img_dict['track_start']) if img_dict['breakdown']>0 else -1

        formation_label = self.create_smoothed_label([formation_idx], n_timepoints) if formation_idx >= 0 else np.zeros(n_timepoints)
        breakdown_label = self.create_smoothed_label([breakdown_idx], n_timepoints, sigma = 1.0) if breakdown_idx >= 0 else np.zeros(n_timepoints)

        interphase_start = formation_idx + 1 if formation_idx >= 0 else 0
        interphase_stop = breakdown_idx - 1 if breakdown_idx >= 0 else n_timepoints-1
        interphase_label = self.create_smoothed_label([interphase_start, interphase_stop], n_timepoints)



        # assumes first and last timepoints are mitotic
        mitotic_coords = []
        if formation_idx > 0:
            mitotic_coords += [0, formation_idx-1]
        if breakdown_idx >= 0 and breakdown_idx < (n_timepoints-1):
            mitotic_coords += [breakdown_idx+1, n_timepoints-1]
    
        mitotic_label= self.create_smoothed_label(mitotic_coords, n_timepoints) if len(mitotic_coords)>0 else np.zeros(n_timepoints)

        if breakdown_idx >=0:
            interphase_label[breakdown_idx-1:] = 0

        # 0: normal, 1: formation, 2: breakdown, 3: mitotic
        label = np.stack([
            interphase_label,
            formation_label,
            breakdown_label,
            mitotic_label,
        ]).T
        img_dict['label'] = label

        return img_dict


class GenerateLabels(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, img_dict):
        n_timepoints = img_dict['img'].shape[0]
        formation_idx = int(img_dict['formation'] - img_dict['track_start'])
        breakdown_idx = int(img_dict['breakdown'] - img_dict['track_start'])

        # 0: normal, 1: formation, 2: breakdown 3: mitotic
        tp_labels = np.zeros(n_timepoints)
        if 0 <=formation_idx < len(tp_labels):
            tp_labels[:formation_idx] = 1
            # tp_labels[formation_idx] = 1

        if 0 <= breakdown_idx < len(tp_labels):
            tp_labels[breakdown_idx+1:] = 1
            # tp_labels[breakdown_idx] = 2
        img_dict['label'] = tp_labels
        return img_dict


from scipy.ndimage import gaussian_filter
class GenerateLabelsRegression(Transform):
    def __init__(self, max_track_length=250):
        super().__init__()
        self.max_track_length = max_track_length

    def create_smoothed_label(self, idx, n_timepoints):
        label = np.zeros(self.max_track_length + 1)
        if idx > 0:
            label[idx] = 1
            # label smoothing
            label = gaussian_filter(label, sigma=0.8)
            label /= label.max()
            # last index is for -1 = no formation/breakdown
            label[n_timepoints:] = 0 
        else:
            label[-1] = 1
        return label

    def __call__(self, img_dict):
        n_timepoints = img_dict['img'].shape[0]
        formation_idx = int(img_dict['formation'] - img_dict['track_start'])
        breakdown_idx = int(img_dict['breakdown'] - img_dict['track_start'])

        # 0: normal, 1: mitotic
        tp_labels = np.zeros(n_timepoints)
        if 0 <=formation_idx < len(tp_labels):
            tp_labels[:formation_idx] = 1
        else:
            formation_idx = -10

        if 0 <= breakdown_idx < len(tp_labels):
            tp_labels[breakdown_idx+1:] = 1
        else:
            breakdown_idx = -10

        # tp_labels[n_timepoints:] = 2
        img_dict['tp_label'] = tp_labels

        img_dict['window_labels'] = torch.Tensor([formation_idx, breakdown_idx])
        
        # img_dict['breakdown_label'] = self.create_smoothed_label(breakdown_idx, n_timepoints)
        # img_dict['formation_label'] = self.create_smoothed_label(formation_idx, n_timepoints)

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
    
    def __call__regression(self, img_dict):
        new_im_dict = copy.deepcopy(img_dict)
        new_im = img_dict['img'].copy()
        window_labels = img_dict['window_labels'].clone()


        n = self.R.randint(*(self.percentage*new_im.shape[0]).astype(int), size = 1)[0]
        n = max(20, n)
        #crop to start of track
        if self.R.random() < self.p_first:
            new_im_dict['img'] = new_im[:n]
            new_im_dict['tp_label'] = new_im_dict['tp_label'][:n]
            if window_labels[1]>= n:
                window_labels[1] = -10
                new_im_dict['window_labels'] = window_labels
        # crop to end of track
        elif self.R.random() < self.p_last:
            new_im_dict['img'] = new_im[-n:]
            new_im_dict['tp_label'] = new_im_dict['tp_label'][-n:]
            # adjust breakdown index
            window_labels[1] -= n
            if window_labels[0] < n:
                # no formation
                window_labels[0] = -10
            new_im_dict['window_labels'] = window_labels
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

from monai.transforms import Resize

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


class WindowCrop(RandomizableTransform):
    """
    Classify window + specify index where transition occurs 
    """
    def __init__(
        self,
        window_size: int= 3,
        num_samples: int = 16,
        reweight: bool = True
    ):
        """
        Parameters
        ----------

        """
        super().__init__()
        self.window_size = window_size
        self.reweight = reweight
        self.num_samples = num_samples


    def __call__(self, img_dict):
        n_timepoints = img_dict['img'].shape[0]
        formation_idx = int(img_dict['formation'] - img_dict['track_start'])
        breakdown_idx = int(img_dict['breakdown'] - img_dict['track_start'])
        #give equal probability to formation, breakdown,and normal timepoints
        p = np.ones(n_timepoints)
        if self.reweight:
            if formation_idx >= 0 and formation_idx < n_timepoints:
                p[formation_idx] =  n_timepoints * 2
            if breakdown_idx < n_timepoints and breakdown_idx >= 0:
                p[breakdown_idx] =   n_timepoints
        p = p[:-self.window_size]
        # upweight timepoints next to formation/breakdown
        p = gaussian(p)
        p /= p.sum()

        if self.num_samples > 0:
            window_start = self.R.choice(n_timepoints - self.window_size, size=self.num_samples , p = p)
        else:
            window_start = np.arange(len(p))
        window_end = window_start + self.window_size
        
        # 0: normal, 1: formation, 2: breakdown 3: mitotic
        tp_labels = np.zeros(n_timepoints)
        if 0 <=formation_idx < len(tp_labels):
            tp_labels[:formation_idx] = 3
            tp_labels[formation_idx] = 1

        if 0 <= breakdown_idx < len(tp_labels):
            tp_labels[breakdown_idx+1:] = 3
            tp_labels[breakdown_idx] = 2

        data =[]

        for ws, we in zip(window_start, window_end):
            #label if transition occurs in window
            label = 0
            if  ws < formation_idx < we:
                label = 1
            elif ws <= breakdown_idx < we-1:
                label = 2

            data.append({'img': img_dict['img'][ws:we] , 'label': label, 'tp_label': tp_labels[ws:we] })

        return data

