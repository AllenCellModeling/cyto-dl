from cyto_dl.nn.head import BaseHead
from aicsimageio.writers import OmeTiffWriter

class MAEHead(BaseHead): 
    def save_image(self, im, pred, mask):
        out_path = self.filename_map["output"][0]
        
        y_hat_out = self._postprocess(pred[0], img_type='prediction')
        OmeTiffWriter.save(data=y_hat_out, uri=out_path)

        y_out = self._postprocess(im[0], img_type="input")
        OmeTiffWriter.save(data=y_out, uri=str(out_path).replace(".t", "_input.t"))
        
        OmeTiffWriter.save(data=mask, uri=str(out_path).replace(".t", "_mask.t"))

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

        # backbone features: {task1: {mask, forward, backward, n_tokens, reconstruction}}
        if run_forward:
            y_hat, mask = backbone_features
        else:
            raise ValueError("MAE head is only intended for use during training.")
        
        loss ={}
        for task in backbone_features:
            y_hat = backbone_features[task]['reconstruction']
            loss = (batch[self.head_name] - y_hat) ** 2
            mask = backbone_features[task]['mask']
            if mask.sum() > 0:
                loss = loss[mask.bool()].mean()
            else:
                loss = loss.mean()
            if save_image:
                self.save_image(im = batch[self.head_name], pred = y_hat, mask = mask)

        loss['loss'] = sum(loss.values())

        return {
            "loss": loss,
            "y_hat_out": None,
            "y_out": None,
        }
