from pathlib import Path
from typing import Callable, Optional, Union, List
import tqdm
import numpy as np
import pandas as pd
from monai.data import DataLoader, Dataset, MetaTensor
from monai.transforms import apply_transform
from aicsimageio import imread, AICSImage
from skimage.transform import resize
import torch

class SingleCellDataset(Dataset):
    """
    Dataset converting nucmorph-style single-cell data into batches of single-cell images. Assumes all passed in images are the same size
    """
    def __init__(
        self,
        csv_path: Union[Path, str],
        seg_path_column: str,
        raw_path_column: str,
        roi_column: str = 'roi',
        seg_out_key: str = 'seg',
        raw_out_key: str = 'raw',
        roi_padding: Optional[List[int]] = [8,20,20],
        roi_size: Optional[List[int]] = [24, 192, 192],
        roi_resize_factor: List[float] = [2.6134, 2.5005,2.5005],
        transform: Optional[Callable] = None,
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
        super().__init__(None, transform)
        self.raw_path_column = raw_path_column
        self.seg_path_column = seg_path_column
        self.raw_out_key = raw_out_key
        self.seg_out_key = seg_out_key
        self.transform = transform

        self.data = pd.read_csv(csv_path)
        self.roi_size = np.array(roi_size)
        self.roi_resize_factor = np.array(roi_resize_factor)
        self.roi_padding = (np.array(roi_padding) / roi_resize_factor).round().astype(int)

        self.raw_image_size = AICSImage(self.data[raw_path_column].iloc[0]).shape[-3:]

        self.data = pd.read_csv(csv_path, usecols=[seg_path_column, raw_path_column, roi_column, 'label_img', 'CellId', 'index_sequence'])
        print('dataframe loaded')

        max_roi_size = np.stack(self.data[roi_column].apply(lambda x: np.array(x[1:-1].split(',')).astype(float)).apply(lambda x: x[1::2] - x[::2])).max(0)
        max_roi_size = (max_roi_size / self.roi_resize_factor).round().astype(int)
        max_roi_size += (max_roi_size % np.array([4, 8, 8])).round().astype(int)
        max_roi_size= np.minimum(max_roi_size, np.array([24, 192, 192]))
        self.max_roi_size = max_roi_size

        self.data['adjusted_rois']= self.data[roi_column].apply(self.validate_roi)
        self.data= self.data.dropna(subset=['adjusted_rois'])

        # self.data= self.data[(self.data.index_sequence%10) == 0]

        self.seg_path_columns= self.data[seg_path_column].unique().tolist()
        print('unique seg paths loaded', len(self.seg_path_columns))
        self.single_cells = []

    def validate_roi_old(self, roi):
        """
        this pads ALL ROIS TO 24x128x128 and results in artifacts where PCs pick up on low-x and low-y because thepositional embeddings used shift
        """
        roi = np.array(roi[1:-1].split(',')).astype(float)
        # rescale from 100x to 20x
        roi[::2] /= self.roi_resize_factor
        roi[1::2] /= self.roi_resize_factor

        # remove padding added by single cell dataset step
        roi[::2] += self.roi_padding
        roi[1::2] -= self.roi_padding

        roi = roi.astype(int)

        # ensure roi is smaller than self.roi_size
        roi_size = roi[1::2] - roi[::2]
        padding = (self.roi_size - roi_size).round().astype(int)

        if np.any(padding < 0):
            raise ValueError(f'ROI {roi_size} is larger than roi_size {self.roi_size}')
        # pad roi while remaning withiin self.seg_image_size
        raw_roi = []
        seg_roi = []
        for i, (start, stop) in enumerate(zip(roi[::2], roi[1::2])):
            # take padding all from start if possible
            start_padding = padding[i] if start - padding[i] >= 0 else start
            padding[i] -= start_padding
            stop_padding = padding[i] if stop + padding[i] <= self.raw_image_size[i] else self.raw_image_size[i] - stop

            start -= start_padding
            stop += stop_padding
            raw_roi.append(slice(start, stop, None))
            seg_roi.append(slice(round(start*self.roi_resize_factor[i]), round(stop*self.roi_resize_factor[i]),  None))
        return tuple(raw_roi), tuple(seg_roi)
    
    def _transform_old(self, index):
        data= self.data[self.data[self.seg_path_column] == self.seg_path_columns.pop()]
        single_cells = self.extract_single_cells(data)
        single_cells = (
            apply_transform(self.transform, single_cells) if self.transform is not None else single_cells
        )
        return [{
            self.raw_out_key: MetaTensor(cell['raw'], meta={"filename_or_obj": cell['cell_id']}),
            self.seg_out_key: MetaTensor(cell['seg'], meta={"filename_or_obj": cell['cell_id']}),
        } for cell in single_cells]
    
    def validate_roi_old2(self, roi):
        """
        This is an improvement, but we can still predict nuclear shape purely from positional embeddings because crop sizes are different
        """
        roi = np.array(roi[1:-1].split(',')).astype(float)
        # rescale from 100x to 20x
        roi[::2] /= self.roi_resize_factor
        roi[1::2] /= self.roi_resize_factor

        # remove padding added by single cell dataset step
        roi[::2] += self.roi_padding
        roi[1::2] -= self.roi_padding
        roi = roi.astype(int)        

        # pad to multiple of patch size
        roi_size = np.array(roi[1::2] - roi[::2])
        padding = roi_size % np.array([4, 8, 8])

        # pad roi while remaning within self.seg_image_size
        raw_roi = []
        seg_roi = []
        for i, (start, stop) in enumerate(zip(roi[::2], roi[1::2])):
            # take padding all from start if possible
            start_padding = padding[i] if start - padding[i] >= 0 else start
            padding[i] -= start_padding
            stop_padding = padding[i] if stop + padding[i] <= self.raw_image_size[i] else self.raw_image_size[i] - stop

            start -= start_padding
            stop += stop_padding
            raw_roi.append(slice(start, stop, None))
            seg_roi.append(slice(round(start*self.roi_resize_factor[i]), round(stop*self.roi_resize_factor[i]),  None))
        return tuple(raw_roi), tuple(seg_roi)
    
    def validate_roi(self, roi):
        """
        This makes all rois the same size and centers them around nuclei
        """
        roi = np.array(roi[1:-1].split(',')).astype(float)
        # rescale from 100x to 20x
        roi[::2] /= self.roi_resize_factor
        roi[1::2] /= self.roi_resize_factor

        # remove padding added by single cell dataset step
        roi[::2] += self.roi_padding
        roi[1::2] -= self.roi_padding
        roi = roi.astype(int)        

        # pad to multiple of patch size
        roi_size = np.array(roi[1::2] - roi[::2])
        padding = self.max_roi_size - roi_size
        if np.any(padding < 0):
            return np.nan
        # pad roi while remaning within self.seg_image_size
        raw_roi = []
        seg_roi = []
        for i, (start, stop) in enumerate(zip(roi[::2], roi[1::2])):
            # calculate padding for start and stop separately
            start_padding = min(padding[i] // 2, start)
            stop_padding = min(padding[i] - start_padding, self.raw_image_size[i] - stop)

            start -= start_padding
            stop += stop_padding
            raw_roi.append(slice(start, stop, None))
            seg_roi.append(slice(round(start*self.roi_resize_factor[i]), round(stop*self.roi_resize_factor[i]),  None))
        return tuple(raw_roi), tuple(seg_roi)
    
    def extract_single_cells(self, subset):
        """
        Load a pair of images from the dataset
        """
        seg_img = imread(subset[self.seg_path_column].iloc[0]).squeeze()
        raw_img = imread(subset[self.raw_path_column].iloc[0]).squeeze()

        raw_img = (raw_img - raw_img.mean()) / raw_img.std()

        single_cells = []
        for row in tqdm.tqdm(subset.itertuples()):
            raw_roi, seg_roi = row.adjusted_rois
            seg = seg_img[seg_roi] == row.label_img
            raw =  raw_img[raw_roi]
            if raw.size == 0:
                continue
            seg = resize(seg, raw.shape, order=0, preserve_range=True)
            single_cells.append({
                'raw':raw[None],
                'seg':seg[None],
                'cell_id':row.CellId,
            })
        return single_cells
    
    def _transform(self, index):
        if len(self.single_cells) == 0:
            # refill when empty
            try:
                data= self.data[self.data[self.seg_path_column] == self.seg_path_columns.pop()]
            except Exception as e:
                print(e)
                return {'stop': torch.ones(1)}
            self.single_cells = self.extract_single_cells(data)

        cell = self.single_cells.pop()
        cell = (
            apply_transform(self.transform, cell) if self.transform is not None else cell
        )
        return {
            self.raw_out_key: MetaTensor(cell['raw'], meta={"filename_or_obj": cell['cell_id']}),
            self.seg_out_key: MetaTensor(cell['seg'], meta={"filename_or_obj": cell['cell_id']}),
        }


    def __len__(self):
        # upper bound on # cells / fov
        return len(self.seg_path_columns) * 1000

def make_single_cell_dataloader(dataset_kwargs, dataloader_kwargs):
    dataset = SingleCellDataset(**dataset_kwargs)
    return DataLoader(dataset, **dataloader_kwargs)

