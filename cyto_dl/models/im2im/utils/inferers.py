from monai.inferers import PatchInferer, Merger
import torch

class EmbeddingPatchMerger(Merger):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.values = []
        self.locations = {}

    def aggregate(self, values, locations):
        """
        Aggregate values for merging.

        Args:
            values: a tensor of shape BCHW[D], representing the values of inference output.
            location: a tuple/list giving the top left location of the patch in the original image.
        """
        self.values.append(values)
        dims = "zyx" if len(locations) == 3 else "yx"
        # cast to string for saving to csv
        for axis, loc in zip(dims, locations):
            try:
                self.locations[f"start_{axis}"].append(loc)
            except KeyError:
                self.locations[f"start_{axis}"] = [loc]

    def finalize(self):
        """
        Finalize the merging process and return the aggregated values.
        Returns:
            Stacked embeddings, shape n_patches x embedding_dim
            Stacked locations, shape n_patches x spatial dims of input
        """
        return torch.cat(self.values, dim=0), self.locations
    
class EmbeddingPatchInferer(PatchInferer):
    """
    This overrides the PatchInferer to allow models that embed a spatial input to a single latent vector to have access to input coordinates of the patch by making the "ratio" between the input and output equal to 1.0 in all dimensions.
    """
    def _initialize_mergers(self, *args, **kwargs):
        mergers, ratios = super()._initialize_mergers(*args, **kwargs)
        ratios = [tuple([1.0] * len(self.splitter.patch_size)) for r in ratios]
        return mergers, ratios
