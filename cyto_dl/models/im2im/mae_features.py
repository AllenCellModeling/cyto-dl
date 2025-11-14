import sys
import warnings
from pathlib import Path
from typing import Dict, List, Union
import numpy as np

import torch
import torch.nn as nn
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference
from torchmetrics import MeanMetric

from cyto_dl.models.base_model import BaseModel

warnings.simplefilter("once", UserWarning)


class MAE_Features(BaseModel):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        task_heads: Dict,
        x_key: str,
        save_dir="./",
        save_images_every_n_epochs=1,
        inference_args: Dict = {},
        inference_heads: Union[List, None] = None,
        compile=False,
        **base_kwargs,
    ):
        """
        Parameters
        ----------
        backbone: nn.Module
            backbone network, parameters are shared between task heads
        task_heads: Dict
            task-specific heads
        x_key: str
            key of input image in batch
        save_dir="./"
            directory to save images during training and validation
        save_images_every_n_epochs=1
            Frequency to save out images during training
        inference_args: Dict = {}
            Arguments passed to monai's [sliding window inferer](https://docs.monai.io/en/stable/inferers.html#sliding-window-inference)
        inference_heads: Union[List, None] = None
            Optional list of heads to run during inference. Defaults to running all heads.
        compile: False
            Whether to compile the model using torch.compile
        **base_kwargs:
            Additional arguments passed to BaseModel
        """

        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
        }

        for head in task_heads.keys():
            assert head not in (
                "loss",
                "pred",
                "target",
                "input",
            ), "Task head name cannot be 'loss', 'pred', 'target', or 'input'"
            _DEFAULT_METRICS.update(
                {
                    f"train/loss/{head}": MeanMetric(),
                    f"val/loss/{head}": MeanMetric(),
                    f"test/loss/{head}": MeanMetric(),
                }
            )

        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.automatic_optimization = True

        if compile is True and not sys.platform.startswith("win"):
            self.backbone = torch.compile(backbone)
            self.task_heads = torch.nn.ModuleDict(
                {k: torch.compile(v) for k, v in task_heads.items()}
            )
        else:
            self.backbone = backbone
            self.task_heads = torch.nn.ModuleDict(task_heads)

        self.inference_heads = inference_heads or list(self.task_heads.keys())

        for k, head in self.task_heads.items():
            head.update_params({"head_name": k, "x_key": x_key, "save_dir": save_dir})

    # def configure_optimizers(self):
    #     opts = []
    #     scheds = []
    #     for key in ("generator", "discriminator"):
    #         if key in self.optimizer.keys():
    #             if key == "generator":
    #                 opt = self.optimizer[key](
    #                     filter(
    #                         lambda p: p.requires_grad,
    #                         list(self.backbone.parameters()) + list(self.task_heads.parameters()),
    #                     )
    #                 )
    #             elif key == "discriminator":
    #                 opt = self.optimizer[key](self.discriminator.parameters())
    #             scheduler = self.lr_scheduler[key](optimizer=opt)
    #             opts.append(opt)
    #             scheds.append(scheduler)
    #     return (opts, scheds)

    def forward(self, x, run_heads):
        z = self.backbone(x)
        return {task: self.task_heads[task](z) for task in run_heads}

    def _inference_forward(self, batch, stage, n_postprocess, run_heads):
        """During inference, we need to calculate per-fov loss/metrics/postprocessing.

        To avoid storing and passing to each head the intermediate results of the backbone, we need
        to run backbone + taskheads patch by patch, then do saving/postprocessing/etc on the entire
        fov.
        """
        with torch.no_grad():
            features = []
            for patch in range(batch[self.hparams.x_key].shape[1]):
                pred = self.forward(batch[self.hparams.x_key][:,patch,...],run_heads)[self.hparams.x_key].to(dtype=torch.float16, device='cpu').numpy().squeeze()
                # import pdb; pdb.set_trace()
                pred = np.mean(pred.reshape((-1,pred.shape[-1])), axis=0).squeeze()

                features.append(pred)
            features = np.stack(features)
        meta = {key:val[0] for key, val in batch.items() if key not in run_heads}
        return (meta, features)
        

    def run_forward(self, batch, stage, n_postprocess, run_heads):
        return self._inference_forward(batch, stage, n_postprocess, run_heads)

    def get_n_postprocess_image(self, batch, batch_idx, stage):
        # save first batch every n epochs during train/val
        if (
            stage in ("train", "val")
            and batch_idx
            == (self.current_epoch + 1) % self.hparams.save_images_every_n_epochs
            == 0
        ):
            return 1
        # postprocess all images in batch for predict/test
        elif stage in ("predict", "test"):
            return batch[self.hparams.x_key].shape[0]
        return 0

    def _sum_losses(self, losses):
        losses["loss"] = torch.sum(torch.stack(list(losses.values())))
        return losses

    def _get_unrun_heads(self, io_map):
        """Returns heads that don't have outputs yet."""
        updated_run_heads = []
        # check that all output files exist for each head
        for head, head_io_map in io_map.items():
            for fn in head_io_map["output"]:
                if not Path(fn).exists():
                    updated_run_heads.append(head)
                    break
        return updated_run_heads

    def _combine_io_maps(self, io_maps):
        """Aggregate io_maps from per-head to per-input image."""
        io_map = {}
        # create input-> per head output mapping
        for head, head_io_map in io_maps.items():
            for in_file, out_file in zip(head_io_map["input"], head_io_map["output"]):
                if in_file not in io_map:
                    io_map[in_file] = {}
                io_map[in_file][head] = out_file
        return io_map

    def _get_run_heads(self, batch, stage, batch_idx):
        """Get heads that are either specified as inference heads and don't have outputs yet or all
        heads."""
        run_heads = self.inference_heads
        if stage in ("train", "val", "test"):
            run_heads = [key for key in self.task_heads.keys() if key in batch]
            return run_heads, None
        filenames = batch[self.hparams.x_key].meta.get("filename_or_obj", None)
        if filenames is None:
            warnings.warn(
                'Batch MetaTensors must have "filename_or_obj" to be saved out. Returning array prediction instead...',
                UserWarning,
            )
            return run_heads, None

        # IO_map is only generated for prediction
        io_map = {h: self.task_heads[h].generate_io_map(filenames) for h in run_heads}
        # only run heads that don't have outputs yet for prediction
        run_heads = self._get_unrun_heads(io_map)
        io_map = self._combine_io_maps(io_map)

        return run_heads, io_map

    def _to_tensor(self, batch):
        """Convert monai metatensors to tensors."""
        for k, v in batch.items():
            if isinstance(v, MetaTensor):
                batch[k] = v.as_tensor()
        return batch

    def get_per_head(self, outs, key):
        return {head_name: head_result[key] for head_name, head_result in outs.items()}

    def model_step(self, stage, batch, batch_idx):
        run_heads, _ = self._get_run_heads(batch, stage, batch_idx)

        n_postprocess = self.get_n_postprocess_image(batch, batch_idx, stage)
        batch = self._to_tensor(batch)
        outs = self.run_forward(batch, stage, n_postprocess, run_heads)
        # aggregate losses across heads
        results = self.get_per_head(outs, "loss")
        results = self._sum_losses(results)

        if n_postprocess > 0:
            # add postprocessed images to return dict
            for k in ("pred", "target", "input"):
                results[k] = self.get_per_head(outs, k)

        self.compute_metrics(results, None, None, stage)

        return results

    def predict_step(self, batch, batch_idx):
        stage = "predict"
        run_heads, io_map = self._get_run_heads(batch, stage, batch_idx)
        outs = None
        if len(run_heads) > 0:
            n_postprocess = self.get_n_postprocess_image(batch, batch_idx, stage)
            
            locs = batch[self.hparams.x_key].meta['location'].cpu().numpy().squeeze()
            shapes = batch[self.hparams.x_key].meta['spatial_shape'].cpu().numpy().squeeze()
            p = np.concatenate((locs,shapes), axis=0)

            rois = [[p[0,i], p[0,i]+p[2,i], p[1,i], p[1,i]+p[3,i]] for i in range(p.shape[1])]
            m = np.max(np.array(rois),axis=0)
            rel_rois = [[r[0]/m[1], r[1]/m[1], r[2]/m[3], r[3]/m[3]] for r in rois]

            vole_args = ['?c1=ven:1&reg={0:.3f}:{1:.3f},{2:.3f}:{3:.3f},0:1&mode=maxproject'.format(r[2],r[3],r[0],r[1]) for r in rel_rois]

            batch = self._to_tensor(batch)
            outs = self.run_forward(batch, stage, n_postprocess, run_heads)
            outs[0]['roi'] = rois
            outs[0]['Link Path'] = vole_args
        
        return io_map, outs
