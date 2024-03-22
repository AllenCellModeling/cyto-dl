import concurrent.futures

import lightning.pytorch as pl

from cyto_dl.models.im2im.utils.instance_seg import InstanceSegCluster


class Segment(pl.Callback):
    def __init__(self, **kwargs):
        super().__init__()
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)
        self.cluster = InstanceSegCluster()
        self.futures = []

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        print(self.futures)
        self.futures = [future for future in self.futures if not future.done()]

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        import time

        t0 = time.time()
        # semantic= (image > 0).cpu()
        # skel =image[0].cpu()
        # embedding= image[2: 5].cpu()

        # future = self.executor.submit(self.cluster, semantic=semantic, skel=skel, embedding=embedding)
        future = self.executor.submit(
            self.cluster,
            image=outputs["nucseg"]["y_hat_out"][0].detach().half().cpu(),
            batch_idx=batch_idx,
        )
        self.futures.append(future)
        print("SUBMITTING FUTURE TAKES", time.time() - t0)
