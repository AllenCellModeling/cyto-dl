from typing import Dict, Union, List

import numpy as np
import torch
from lightning import LightningDataModule
from monai.data import DataLoader, Dataset
from upath import UPath as Path
import tqdm

import pandas as pd
from aicsimageio import AICSImage, imread
from skimage.transform import resize
from multiprocessing.pool import ThreadPool


class TrackClassificationDatamodule(LightningDataModule):
    def __init__(
        self,
        csv_path: Union[Path, str],
        augmentations: Dict,
        num_workers: int = 8,
        splits: List = ['train', 'val'],
        seed: int = 42,
        **dataloader_kwargs,
    ):
        super().__init__()
        torch.manual_seed(seed)

        csv_path = Path(csv_path)

        self.csvs = {}
        self.tracks= {}

        for stage in ("train", "val",'test', "predict"):
            if (csv_path/f'{stage}.csv').exists():
                self.csvs[stage] = pd.read_csv(csv_path/f'{stage}.csv')
                self.tracks[stage] = (self.csvs[stage]['movie'] + '_'+ self.csvs[stage]['track_id'].astype(str)).unique()
            else:
                print(f'WARNING: {csv_path/f"{stage}.csv"} does not exist')

        self.dataloader_kwargs = dataloader_kwargs
        self.augmentations = augmentations
        self.num_workers = num_workers
        self.splits = splits

    def extract_patches(self, df):
        # img = AICSImage(df.data_path.iloc[0]).get_image_dask_data('ZYX',C = 0).compute()
        # img = np.max(img,0).astype(np.float32)
        img = imread(df.data_path.iloc[0]).squeeze().astype(np.float32)
        img = (img - np.mean(img)) / np.std(img)
        data = {}
        #                                        remove [], split on commas, z coords, resize to 20x coords, convert to int
        rois = df['roi'].apply(lambda x: (np.array(x[1:-1].split(',')[2:], dtype=float)/2.5005).astype(int)).values #
        for i, row in enumerate(df.itertuples()):
            roi = rois[i]
            roi[0] = max(0, roi[0] - 25)
            roi[1] = min(img.shape[0], roi[1] + 25)
            roi[2] = max(0, roi[2] - 25)
            roi[3] = min(img.shape[1], roi[3] + 25)
           
            # crop  = img[rois[i][0]:rois[i][1], rois[i][2]:rois[i][3]]
            crop = img[roi[0]:roi[1], roi[2]:roi[3]]
            data[row.track_id] = resize(crop, (64, 64), anti_aliasing=True,preserve_range=True).astype(np.float16)
        data['timepoint'] = df.T_index.iloc[0]
        return data


    def load_movie(self, df):
        data = {}
        pool = ThreadPool(self.num_workers)
        results = []
        for t in sorted(df.T_index.unique()):
            results.append(pool.apply_async(self.extract_patches, kwds={'df':df[df.T_index==t]} ))

        pool.close()
        pool.join()
        data = [r.get() for r in results]

        # iterate by timepoint
        new_data ={}
        for timepoint_patch_dict in data:
            #patch metadata
            timepoint = timepoint_patch_dict['timepoint']
            del timepoint_patch_dict['timepoint']

            # patch data
            for track_id, patch in timepoint_patch_dict.items():
                if track_id not in new_data:
                    new_data[track_id] = {'img': [], 'track_start':int(timepoint), 'track_id':track_id}
                new_data[track_id]['img'].append(patch)
        del data
        datasets = {split: [] for split in self.splits}

        #aggregate by track
        for track_id, data in new_data.items():
            # find split track belongs to 
            for split in self.splits:
                if f'{df.movie.iloc[0]}_{track_id}' in self.tracks[split]:
                    break
            data['img'] = np.stack(data['img'])
            data['track_end'] = int(data['track_start'] + data['img'].shape[0])

            if split != 'predict':
                track_info = df[df.track_id == track_id][['predicted_breakdown', 'predicted_formation']]
                data['breakdown'] = int(track_info.predicted_breakdown.iloc[0])
                data['formation'] = int(track_info.predicted_formation.iloc[0])
            data['movie'] = df.movie.iloc[0]

            datasets[split].append(data)

        del new_data
        return datasets
    
    def prepare_data(self):
        df = pd.concat([self.csvs[split] for split in self.splits])
        datasets = {split: [] for split in self.splits}
        for movie in tqdm.tqdm(df.movie.unique(),desc='Loading movies'):
            d = self.load_movie(df[df.movie==movie])
            print('Movie', movie, 'loaded')
            for split, data in d.items():
                datasets[split].extend(data)
            
        for ds in datasets:
            print('Split', ds,'has ', len(datasets[ds]), 'timepoints')
            datasets[ds] = Dataset(datasets[ds], transform=self.augmentations[ds])
        self.datasets = datasets

    def make_dataloader(self, split):
        kwargs = dict(**self.dataloader_kwargs)
        kwargs["shuffle"] = True
        subset = self.datasets[split]
        return DataLoader(dataset=subset, **kwargs)


    def train_dataloader(self):
        return self.make_dataloader("train")

    def val_dataloader(self):
        return self.make_dataloader("val")

    def test_dataloader(self):
        return self.make_dataloader("test")

    def predict_dataloader(self):
        return self.make_dataloader("predict")
