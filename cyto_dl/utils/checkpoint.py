import torch


def load_checkpoint(model, load_params):
    if load_params.get("weights_only"):
        assert load_params.get(
            "ckpt_path"
        ), "ckpt_path must be provided to with argument weights_only=True"
        # load model from state dict to get around trainer.max_epochs limit, useful for resuming model training from existing weights
        state_dict = torch.load(load_params["ckpt_path"], map_location="cpu")[
            "state_dict"
        ]  # nosec B614
        model.load_state_dict(state_dict, strict=load_params.get("strict", True))
        # set ckpt_path to None to avoid loading checkpoint again with model.fit/model.test
        load_params["ckpt_path"] = None
    elif not load_params.get("strict"):
        raise ValueError("To use `strict=False`, `weights_only` must be set to True")
    return model, load_params
