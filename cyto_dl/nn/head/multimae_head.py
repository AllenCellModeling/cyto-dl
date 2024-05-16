from cyto_dl.nn.head import BaseHead
from aicsimageio.writers import OmeTiffWriter
import numpy as np
from cyto_dl.nn.vits.blocks import CrossAttentionBlock
import torch
from typing import List
from cyto_dl.models.im2im.utils.postprocessing import detach
from einops import rearrange
from cyto_dl.nn.vits.utils import take_indexes
from einops.layers.torch import Rearrange
from monai.networks.blocks import UnetOutBlock, UnetResBlock, UpSample


class OutputAdapter(torch.nn.Module):
    def __init__(
        self,
        base_patch_size: List[int],
        emb_dim: int = 64,
        num_layer: int = 2,
        num_head: int = 4,
        conv: bool = False
    ) -> None:
        super().__init__()
        self.transformer =  torch.nn.ParameterList(
            [
                CrossAttentionBlock(
                    encoder_dim=emb_dim,
                    decoder_dim=emb_dim,
                    num_heads=num_head,
                )
                for _ in range(num_layer)
            ]
        )

        self.decoder_norm = torch.nn.LayerNorm(emb_dim)
        self.head = torch.nn.Linear(emb_dim, torch.prod(torch.as_tensor(base_patch_size))) 
        # self.out_conv = torch.nn.Sequential(UnetResBlock(spatial_dims=3, in_channels=1, out_channels=16, stride=1, kernel_size=3, norm_name='INSTANCE', dropout=0), UnetOutBlock(spatial_dims=3, in_channels=16, out_channels=1, dropout=0)) if conv else torch.nn.Identity()

        # in_channels = 64
        # self.upsample= [] 
        # for i in range(3):
        #     self.upsample += [
        #         UpSample(
        #             spatial_dims=3,
        #             in_channels=in_channels,
        #             out_channels=in_channels // 2,
        #             scale_factor= [2,2,2]
        #         )
        #     ]
        #     in_channels //= 2

        # self.upsample = torch.nn.Sequential(*self.upsample)

        # self.out_conv = torch.nn.Sequential(UnetResBlock(spatial_dims=3, in_channels=in_channels, out_channels=16, stride=1, kernel_size=3, norm_name='INSTANCE', dropout=0), UnetOutBlock(spatial_dims=3, in_channels=16, out_channels=1, dropout=0)) if conv else torch.nn.Identity()

    def conv(self, img):
        return img
        # img = self.upsample(img)
        # img = torch.nn.functional.upsample(img, scale_factor=(1.5,3, 3), mode='trilinear', align_corners=False)
        return self.out_conv(img)

    def forward(self, masked, visible):
        for i, transformer in enumerate(self.transformer):
            masked = transformer(masked, visible[i])
        return masked
    
    def output(self, masked):
        return self.head(self.decoder_norm(masked))
    

class MultiMAEHead(BaseHead): 
    def __init__(
        self,
        base_patch_size: List[int],
        num_patches: List[int],
        input_key: str = None,
        emb_dim: int = 64,
        num_layer: int = 2,
        num_head: int = 2,
        spatial_dims: int = 3,
        conv: bool = False,
        mode: str = 'mae',
        loss = torch.nn.MSELoss(reduction="none"),
        postprocess={"input": detach, "prediction": detach},
        save_input=False,
    ):
        """
        Parameters
        ----------
        loss
            Loss function for task
        postprocess={"input": detach, "prediction": detach}
            Postprocessing for `input` and `predictions` of head
        save_input=False
            Whether to save out example input images during training
        """
        super().__init__(loss=loss, postprocess=postprocess, save_input=save_input)

        self.input_key = input_key
        self.mode= mode

        self.model = OutputAdapter(
            base_patch_size=base_patch_size,
            emb_dim=emb_dim,
            num_layer=num_layer,
            num_head=num_head,
            conv=conv
        )

        if spatial_dims == 3:
            self.patch2img = Rearrange(
                "(n_patch_z n_patch_y n_patch_x) b (c patch_size_z patch_size_y patch_size_x) ->  b c (n_patch_z patch_size_z) (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
                n_patch_z=num_patches[0],
                n_patch_y=num_patches[1],
                n_patch_x=num_patches[2],
                patch_size_z=base_patch_size[0],
                patch_size_y=base_patch_size[1],
                patch_size_x=base_patch_size[2],
            )
        elif spatial_dims == 2:
            self.patch2img = Rearrange(
                "(n_patch_y n_patch_x) b (c patch_size_y patch_size_x) ->  b c (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
                n_patch_y=num_patches[0],
                n_patch_x=num_patches[1],
                patch_size_y=base_patch_size[0],
                patch_size_x=base_patch_size[1],
            )
        
    def save_image(self, im, pred, label, mask):
        out_path = self.filename_map["output"][0]
        
        y_hat_out = self._postprocess(pred[0], img_type='prediction')
        OmeTiffWriter.save(data=y_hat_out, uri=out_path)

        y_out = self._postprocess(im[0], img_type="input")
        OmeTiffWriter.save(data=y_out, uri=str(out_path).replace(".t", "_input.t"))

        OmeTiffWriter.save(data=detach(label[0]).astype(np.uint8), uri=str(out_path).replace(".t", "_label.t"))
        OmeTiffWriter.save(data=detach(mask[0]).astype(np.uint8), uri=str(out_path).replace(".t", "_mask.t"))


    def mae_forward(self, task_info, visible_tokens):
        # paper uses task + mask tokens as cross attention queries to everything else, can we do cross mae by just doing mask tokens to everything? then we don't have to separate everything out
        # they also don't remove the task tokens from the contextembeddings so the "cross attention" layer is also doing self attention to the visual tokens of each task basically

        # cross attention between masked tokens of each task and visible tokens of all tasks

        mask_tokens= task_info['masked_tokens']
        mask_tokens = rearrange(mask_tokens, "t b c -> b t c")
        visible_tokens = rearrange(visible_tokens, "n t b c -> n b t c")

        mask_tokens = self.model(mask_tokens, visible_tokens)

        # noneed to remove cls token, it's a part of the visible tokens
        mask_tokens = rearrange(mask_tokens, "b t c -> t b c")

        # (npatches x npatches x npatches) b (emb dim) -> (npatches* npatches * npatches) b (z y x)
        patches = self.model.output(mask_tokens)

        # add back in visible/encoded tokens that we don't calculate loss on
        patches = torch.cat(
                            # tokens                batch size         embedding dimension
            [torch.zeros((task_info['n_tokens'], patches.shape[1], patches.shape[-1]), requires_grad=False, device=patches.device), patches],
            dim=0,
        )
        patches = take_indexes(patches, task_info['backward_indices'])

        task_info['reconstruction'] = self.patch2img(patches)
        return task_info
    
    def finetune_forward(self, task_info, visible_tokens):
        task_tokens = task_info['visible_tokens'][-1]
        task_tokens = rearrange(task_tokens, "t b c -> b t c")
        visible_tokens = rearrange(visible_tokens, "n t b c -> n b t c")

        # cross attention from task-specific tokens to all visible tokens. 
        # in the one_task case, this is just self attention
        task_tokens = self.model(task_tokens, visible_tokens)

        task_tokens = rearrange(task_tokens, "b t c -> t b c")
        # remove cls token
        task_tokens = task_tokens[1:]

        # (npatches x npatches x npatches) b (emb dim) -> (npatches* npatches * npatches) b (z y x)
        patches = self.model.output(task_tokens)
        patches = take_indexes(patches, task_info['backward_indices'])

        task_info['reconstruction'] = self.model.conv(self.patch2img(patches))
        return task_info

    def forward(self, task_info, visible_tokens):
        if self.mode == 'mae':
            return self.mae_forward(task_info, visible_tokens)
        else:
            return self.finetune_forward(task_info, visible_tokens)

    def run_head(
        self,
        backbone_features,
        batch,
        stage,
        save_image,
        run_forward=True,
        y_hat=None,
    ):
        """Run head on backbone features, calculate loss, postprocess and save image, and calculate
        metrics."""
        if not run_forward:
            raise ValueError("MAE head is only intended for use during training.")
        # if input_key is not provided (e.g. during mae pretraining), use head_name as input.
        input_key = self.input_key or self.head_name

        task_info = backbone_features[input_key]
        visible_tokens= backbone_features['visible_tokens']
        del backbone_features

        task_info = self.forward(task_info, visible_tokens)

        y_hat = task_info['reconstruction']

        task_loss = self.loss(y_hat, batch[self.head_name])
        mask = task_info['mask']
        if mask.sum() > 0:
            task_loss = task_loss[mask.bool()].mean()
        else:
            task_loss = task_loss.mean()
        if save_image:
            self.save_image(im = batch[input_key], pred = y_hat, label = batch[self.head_name], mask = mask)
        return {
            "loss": task_loss,
            "y_hat_out": None,
            "y_out": None,
        }
