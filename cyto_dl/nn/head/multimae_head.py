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


class OutputAdapter(torch.nn.Module):
    def __init__(
        self,
        base_patch_size: List[int],
        emb_dim: int = 64,
        num_layer: int = 2,
        num_head: int = 4,
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
        emb_dim: int = 64,
        num_layer: int = 2,
        num_head: int = 4,
        spatial_dims: int = 3,
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
        super().__init__(loss=None)
        self.postprocess = postprocess

        self.model = OutputAdapter(
            base_patch_size=base_patch_size,
            emb_dim=emb_dim,
            num_layer=num_layer,
            num_head=num_head,
        )
        self.save_input = save_input

                
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
        
    def save_image(self, im, pred, mask):
        out_path = self.filename_map["output"][0]
        
        y_hat_out = self._postprocess(pred[0], img_type='prediction')
        OmeTiffWriter.save(data=y_hat_out, uri=out_path)

        y_out = self._postprocess(im[0], img_type="input")
        OmeTiffWriter.save(data=y_out, uri=str(out_path).replace(".t", "_input.t"))
        
        OmeTiffWriter.save(data=mask[0].detach().cpu().numpy().astype(np.uint8), uri=str(out_path).replace(".t", "_mask.t"))


    def forward(self, task_info, visible_tokens):
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
            [torch.zeros((task_info['n_tokens'], patches.shape[1], patches.shape[-1]), requires_grad=False).to(patches), patches],
            dim=0,
        )
        patches = take_indexes(patches, task_info['backward_indices'])

        task_info['reconstruction'] = self.patch2img(patches)
        return task_info

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
        task_info = backbone_features[self.head_name]
        visible_tokens= backbone_features['visible_tokens']
        del backbone_features

        task_info = self.forward(task_info, visible_tokens)

        y_hat = task_info['reconstruction']
        task_loss = (batch[self.head_name] - y_hat) ** 2
        mask = task_info['mask']
        if mask.sum() > 0:
            task_loss = task_loss[mask.bool()].mean()
        else:
            task_loss = task_loss.mean()
        if save_image:
            self.save_image(im = batch[self.head_name], pred = y_hat, mask = mask)
        return {
            "loss": task_loss,
            "y_hat_out": None,
            "y_out": None,
        }
