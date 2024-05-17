from pathlib import Path
from typing import Callable, Optional, Union
import pandas as pd
from monai.data import  Dataset
from monai.transforms import apply_transform
import numpy as np

class NeighborDataset(Dataset):
    """
    Dataset returning neighboring images in tracks from nucmorph-style dataset 
    """
    def __init__(
        self,
        csv_path: Union[Path, str],
        path_column: str = 'path',
        track_id_column: str = 'track_id',
        timepoint_column: str = 'index_sequence',
        max_timepoint_gap: int= 1,
        transform: Optional[Callable] = None,
        **persistent_args,
    ):
        """
        Parameters
        ----------
        csv_path: Union[Path, str]
            path to csv
        img_path_column: str
            column in `csv_path` that contains path to CZI file
        out_key:str
            Key where single-scene/timepoint/channel is saved in output dictionary
        transform: Optional[Callable] = None
            Callable to that accepts numpy array. For example, image normalization functions could be passed here.
        dask_load: bool = True
            Whether to use dask to load images. If False, full images are loaded into memory before extracting specified scenes/timepoints.
        """
        super().__init__(pd.read_csv(csv_path), transform, **persistent_args)
        # subsample to only a few track_ids

        # check tracks at end of movies
        self.data= self.data[(100 < self.data.index_sequence) & (self.data.index_sequence < 570)]

        self.data = self.data[self.data[track_id_column].isin(np.random.choice(self.data[track_id_column].unique(), size=100, replace=False))]
        self.path_column = path_column
        self.track_id_column = track_id_column
        self.timepoint_column = timepoint_column
        self.max_timepoint_gap = max_timepoint_gap
    
    def try_negative(self, row, idx):
        negative = self.data[(self.data[self.track_id_column] == row[self.track_id_column]) & (self.data[self.timepoint_column] - idx <= self.max_timepoint_gap ) & (self.data[self.timepoint_column] - idx > 0)]

        if negative.shape[0] > 0:
            negative = negative.sample(1)
            return negative
        return None

    def _transform(self, index):
        row = self.data.iloc[index]
        idx = row[self.timepoint_column]
        negative = self.try_negative(row, idx)

        while negative is None:
            index = (index + 1) % len(self.data)

            row = self.data.iloc[index]
            idx = row[self.timepoint_column]
            negative = self.try_negative(row, idx)

        data = {
            'anchor': row[self.path_column],
            'negative': negative[self.path_column].values[0],
            'cell_id': row['CellId'],
        }

        return  apply_transform(self.transform, data)
    

from aicsimageio import AICSImage, imread
import time
import tqdm
from multiprocessing import Pool
class NeighborDatasetFull(Dataset):
    """
    Dataset returning neighboring images in tracks from nucmorph-style dataset 
    """
    def __init__(
        self,
        split='train',
        steps_per_epoch=1000,
        transform: Optional[Callable] = None,
    ):

        paths = pd.read_csv("//allen/aics/assay-dev/users/Benji/hydra_workflow/reprocess_parallel_vit_small/normal_scene4/4/metadata.csv", usecols=['T', 'split_image/split_image'])

        
        # img = AICSImage('/allen/programs/allencell/data/proj0/935/c72/962/3cc/7a4/37b/f4f/6d8/69c/91a/71/20200323_F01_001.czi')
        # scene 4 is mama bear
        # img.set_scene(4)
        t0 = time.time()
        timepoints = list(range(50)) if split == 'train' else list(range(100, 110))
        print(split, timepoints)
        paths = paths[paths['T'].isin(timepoints)]
        paths.sort_values('T', inplace=True)
        # img = img.get_image_dask_data('TZYX', C=0, T= timepoints).compute()

        with Pool(32) as p:
            img = p.map(self.load, paths['split_image/split_image'])

        super().__init__(img, transform)
        self.steps_per_epoch = steps_per_epoch
        self.timepoints = len(img)

    def load(self, path):
        print('start')
        img = imread(path).squeeze()
        img = (img-img.mean())/img.std()
        print('end')
        return img

    def _transform(self, index):
        timepoint_start = index % (self.timepoints-1)

        data = {
            'anchor': self.data[timepoint_start][None],
            'negative': self.data[timepoint_start+1][None]
        }

        return  apply_transform(self.transform, data)
    
    def __len__(self):
        return self.steps_per_epoch