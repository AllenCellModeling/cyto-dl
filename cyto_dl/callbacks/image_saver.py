from lightning.pytorch.callbacks import Callback
from pathlib import Path
from bioio.writers import OmeTiffWriter
from typing import List, Union


class ImageSaver(Callback):
    def __init__(self, save_dir: Union[str, Path], save_every_n_epochs:int=1, stages:List[str]=["train", "val"], save_input:bool =False):
        """
        Callback for saving images after postprocessing by eads.

        Parameters
        ----------
        save_dir: Union[str, Path]
            Directory to save images
        save_every_n_epochs:int=1
            Frequency to save images
        stages:List[str]=["train", "val"]
            Stages to save images
        save_input:bool =False
            Whether to save input images
        """
        self.save_dir = Path(save_dir)
        for stage in stages:
            assert stage in ('train', 'val', 'test', 'predict')
            (self.save_dir/f"{stage}_images").mkdir(exist_ok=True, parents=True)
        self.save_every_n_epochs = save_every_n_epochs
        self.stages = stages
        self.save_keys = ['pred', 'target']
        if save_input:
            self.save_keys.append('input')

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if 'predict' in self.stages:
            io_map, outputs = outputs
            if outputs is None:
                # image has already been saved
                return
            for i, head_io_map in enumerate(io_map.values()):
                for k, save_path in head_io_map.items():
                    OmeTiffWriter.save(uri=save_path, data = outputs[k]['pred'][i])

    # train/test/val
    def save(self, outputs, stage=None, step=None):
        for k in self.save_keys:
            for head in outputs[k]:
                fn = self.save_dir / f'{stage}_images/{step}_{head}_{k}.tif'
                OmeTiffWriter.save(uri=fn, data=outputs[k][head])


    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if 'test' in self.stages:
            # save all test outputs
            self.save(outputs, "test", trainer.global_step)

    def _should_save(self, batch_idx, epoch):
        return batch_idx == 0 and (epoch + 1) % self.save_every_n_epochs == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if 'train' in self.stages and self._should_save(batch_idx, trainer.current_epoch):
            self.save(outputs, "train", trainer.global_step)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if 'val' in self.stages and self._should_save(batch_idx, trainer.current_epoch):
            self.save(outputs, "val", trainer.global_step)