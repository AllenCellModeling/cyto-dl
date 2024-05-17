import torch
import torch.nn as nn
import torch.nn.functional
from torchmetrics import MeanMetric
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from cyto_dl.models.base_model import BaseModel
from pathlib import Path
import pandas as pd
import numpy as np

class Triplet(BaseModel):
    def __init__(
        self,
        save_dir,
        *,
        model: nn.Module,
        **base_kwargs,
    ):
        """
        Parameters
        ----------
        model: nn.Module
            model network, parameters are shared between task heads
        x_key: str
            key of input image in batch
        save_dir="./"
            directory to save images during training and validation
        save_images_every_n_epochs=1
            Frequency to save out images during training
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
        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.model = model
        self.loss_fn = torch.nn.TripletMarginLoss(margin=1.0)


    def forward(self, x1, x2):
        return self.model(x1, x2)
    

    def plot_classes(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        # calculate pca on predictions and label by labels
        pca = PCA(n_components=2)
        pca.fit(anchor_embeddings)
        
        random_examples = np.random.choice(anchor_embeddings.shape[0], 10)
        anchor_embeddings = pca.transform(anchor_embeddings)[random_examples]
        positive_embeddings = pca.transform(positive_embeddings)[random_examples]
        negative_embeddings = pca.transform(negative_embeddings)[random_examples]

        fig, ax = plt.subplots()

        # plot anchor embeddings in gray
        ax.scatter(anchor_embeddings[:, 0], anchor_embeddings[:, 1], c='gray')

        # plot positive embeddings in green
        ax.scatter(positive_embeddings[:, 0], positive_embeddings[:, 1], c='green')

        # plot negative embeddings in red
        ax.scatter(negative_embeddings[:, 0], negative_embeddings[:, 1], c='red')


        # draw lines between anchor and positive, anchor and negative
        ax.plot([anchor_embeddings[:, 0], positive_embeddings[:, 0]], [anchor_embeddings[:, 1], positive_embeddings[:, 1]], 'green')
        ax.plot([anchor_embeddings[:, 0], negative_embeddings[:, 0]], [anchor_embeddings[:, 1], negative_embeddings[:, 1]], 'red', alpha=0.1)


        fig.savefig(Path(self.hparams.save_dir) / f"{self.current_epoch}_pca.png")
        plt.close(fig)


    def model_step(self, stage, batch, batch_idx):
        anchor_embeddings = self.model(batch["anchor"])
        anchor_embeddings= torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)

        positive_embeddings = self.model(batch['positive'])
        positive_embeddings= torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)

        negative_embeddings = self.model(batch['negative'])
        negative_embeddings= torch.nn.functional.normalize(negative_embeddings, p=2, dim=1)

        # find triplet loss
        loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

        if stage == 'val' and batch_idx == 0:
            with torch.no_grad():
                self.plot_classes(anchor_embeddings.detach().cpu().numpy(), positive_embeddings.detach().cpu().numpy(), negative_embeddings.detach().cpu().numpy())

        return loss, None, None
    
    # def predict_step(self, batch, batch_idx):
    #     from monai import transforms
    #     import tqdm

    #     cell_ids = batch['cell_id']
    #     anchor = batch['anchor']
    #     embeddings_anchor = self.model(anchor).detach().cpu().numpy()


    #     embeddings_anchor = pd.DataFrame(embeddings_anchor, columns = [str(i) for i in range(embeddings_anchor.shape[1])])
    #     embeddings_anchor['cell_id'] = cell_ids
    #     embeddings_anchor['name'] = 'anchor'

    #     embeddings_negative = self.model(batch['negative']).detach().cpu().numpy()
    #     embeddings_negative = pd.DataFrame(embeddings_negative, columns = [str(i) for i in range(embeddings_negative.shape[1])])
    #     embeddings_negative['cell_id'] = cell_ids
    #     embeddings_negative['name'] = 'negative'


    #     positive_embeddings = []
    #     name = []
    #     cellid = []
    #     # augs = []
    #     for img in tqdm.tqdm(range(anchor.shape[0])):
    #         aug = anchor[img].clone()
    #         for i in range(100):
    #             i_aug = transforms.RandGridDistortion(prob=1)(aug)
    #             # augs.append(i_aug.detach().cpu().numpy())
    #             positive_embeddings.append(self.model(i_aug.unsqueeze(0)).detach().cpu().numpy())
    #             name += [f"grid_distort_{i}"]
    #             cellid += [cell_ids[img]]

    #         for std in np.linspace(0.1, 3.0, 5):
    #             for i in range(100):
    #                 i_aug = transforms.RandGaussianNoise(prob=1, std=std)(aug)
    #                 # augs.append(i_aug.detach().cpu().numpy())
    #                 positive_embeddings.append(self.model(i_aug.unsqueeze(0)).detach().cpu().numpy())
    #                 name += [f"gaussian_noise_{std}"]
    #                 cellid += [cell_ids[img]]

    #         for i in range(4):
    #             i_aug = transforms.Rotate90(k=i, spatial_axes=(1, 2))(aug)
    #             positive_embeddings.append(self.model(i_aug.unsqueeze(0)).detach().cpu().numpy())
    #             name += [f"rotate_90_{i}"]
    #             cellid += [cell_ids[img]]
    #         for hflip in [True, False]:
    #             i_aug = transforms.Flip(spatial_axis=1)(aug)
    #             positive_embeddings.append(self.model(i_aug.unsqueeze(0)).detach().cpu().numpy())
    #             name += [f"hflip_{hflip}"]
    #             cellid += [cell_ids[img]]
    #         for vflip in [True, False]:
    #             i_aug = transforms.Flip(spatial_axis=2)(aug)
    #             positive_embeddings.append(self.model(i_aug.unsqueeze(0)).detach().cpu().numpy())
    #             name += [f"vflip_{vflip}"]
    #             cellid += [cell_ids[img]]
    #         for _ in range(100):
    #             i_aug  = transforms.RandHistogramShift(prob=1, num_control_points=(80, 120))(aug)
    #             positive_embeddings.append(self.model(i_aug.unsqueeze(0)).detach().cpu().numpy())
    #             name += [f"intensity"]
    #             cellid += [cell_ids[img]]

    #     # create csv with batch x embeddings and cell_ids
    #     positive_embeddings = np.stack(positive_embeddings).squeeze(1)
    #     positive_embeddings = pd.DataFrame(positive_embeddings, columns = [str(i) for i in range(positive_embeddings.shape[1])])
    #     positive_embeddings['cell_id'] = cellid
    #     positive_embeddings['name'] = name


    #     all = pd.concat([embeddings_anchor, embeddings_negative, positive_embeddings])
    #     all.to_csv(Path(self.hparams.save_dir) / f"{batch_idx}_embeddings.csv", index=False)
        
        # from aicsimageio.writers import OmeTiffWriter
        # OmeTiffWriter.save(uri = Path(self.hparams.save_dir) / f"{batch_idx}_aug.ome.tiff", data = np.stack(augs), dimension_order='CZYX')

        # OmeTiffWriter.save(uri = Path(self.hparams.save_dir) / f"{batch_idx}_anchor.ome.tiff", data = anchor.detach().cpu().numpy(), dimension_order='CZYX')

        # OmeTiffWriter.save(uri = Path(self.hparams.save_dir) / f"{batch_idx}_negative.ome.tiff", data = batch['negative'].detach().cpu().numpy(), dimension_order='CZYX')

        # quit()

        



    
    def predict_step(self, batch, batch_idx):
        cell_ids = batch['cell_id']
        embeddings = self.model(batch['anchor']).detach().cpu().numpy()

        # create csv with batch x embeddings and cell_ids
        data = pd.DataFrame(embeddings, columns = [str(i) for i in range(embeddings.shape[1])])
        data['cell_id'] = cell_ids
        data.to_csv(Path(self.hparams.save_dir) / f"{batch_idx}_embeddings.csv", index=False)

