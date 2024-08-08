import torch


class EncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        encoder_ckpt=None,
        decoder_ckpt=None,
        train_encoder=True,
        train_decoder=True,
        latent_norm=None,
    ):
        super().__init__()

        self.encoder = self._init_model(encoder, encoder_ckpt, train_encoder)
        self.decoder = self._init_model(decoder, decoder_ckpt, train_decoder)

        self.latent_norm = latent_norm if latent_norm is not None else torch.nn.Identity()

    def _init_model(self, model, ckpt, train):
        if ckpt is not None:
            state_dict = torch.load(ckpt)["state_dict"]
            state_dict = {
                k.replace("backbone.", ""): v
                for k, v in state_dict.items()
                if "net." in k or "final" in k
            }
            model.load_state_dict(state_dict)
        if not train:
            for param in model.parameters():
                param.requires_grad = False
        return model

    def forward(self, x):
        feats = self.encoder(x)
        feats = self.latent_norm(feats)
        return self.decoder(feats)
