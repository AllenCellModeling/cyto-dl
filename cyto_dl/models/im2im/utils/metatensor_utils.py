from monai.data.meta_tensor import MetaTensor

def metatensor_batch_to_tensor(batch):
    """Convert monai metatensors to tensors."""
    for k, v in batch.items():
        if isinstance(v, MetaTensor):
            batch[k] = v.as_tensor()
    return batch