import torch
from monai.inferers import Merger, PatchInferer


class EmbeddingPatchMerger(Merger):
    def __init__(self, spatial_dims: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.values = []
        if spatial_dims not in (2, 3):
            raise ValueError(f"Expected spatial_dims to be 2 or 3, got {spatial_dims}")
        self.spatial_dims = spatial_dims
        self.dim_names = ["z", "y", "x"][-spatial_dims:]
        self.locations = {f"start_{ax}": [] for ax in self.dim_names}

    def aggregate(self, values, locations):
        """Aggregate values for merging.

        Args:
            values: a tensor of shape BC[Z]YX, representing the values of inference output.
            location: a tuple/list giving the top left location of the patch in the original image.
        """
        b = values.shape[0]
        self.values.append(values)
        if len(locations) != self.spatial_dims:
            raise ValueError(
                f"Expected {self.spatial_dims} spatial dimensions, got {len(locations)}"
            )
        # cast to string for saving to csv
        for axis, loc in zip(self.dim_names, locations):
            loc = [loc] * b if b > 1 else loc
            self.locations[f"start_{axis}"].append(loc)

    def finalize(self):
        """Finalize the merging process and return the aggregated values.

        Returns:
            Stacked embeddings, shape n_patches x embedding_dim
            Stacked locations, shape n_patches x spatial dims of input
        """
        return torch.cat(self.values, dim=0), self.locations


class EmbeddingPatchInferer(PatchInferer):
    """This overrides the PatchInferer to allow models that embed a spatial input to a single
    latent vector to have access to input coordinates of the patch.

    The typical use of the PatchInferer is for image-to-image applications, where the input and
    output are both spatial but might have different spatial sizes (e.g. in superresolution the
    output it larger than the input). The ratio is used to calculate the location of an output
    patch in the input image, so making the "ratio" between the input and output 1.0 in all
    dimensions associates each latent dimension with the patch it came from.
    """

    def _initialize_mergers(self, *args, **kwargs):
        mergers, ratios = super()._initialize_mergers(*args, **kwargs)
        ratios = [tuple([1.0] * len(self.splitter.patch_size)) for r in ratios]
        return mergers, ratios
