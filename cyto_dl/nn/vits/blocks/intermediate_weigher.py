import torch
from einops import rearrange


class IntermediateWeigher(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, n_outputs, norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.weights = torch.nn.Linear(num_layers, n_outputs)
        # initialize with equal weighting of all layers
        self.weights.weight.data.fill_(1.0 / num_layers)
        self.weights.bias.data.zero_()

        self.norms = torch.nn.ModuleList([norm_layer(embed_dim) for _ in range(num_layers)])

    def forward(self, x):
        """Apply layer norm to each intermediate feature and return n_outputs weighted sums, last
        dimension is n_outputs."""
        x = torch.stack([norm(x[i]) for i, norm in enumerate(self.norms)], dim=-1)
        x = self.weights(x)
        x = rearrange(x, " b t c n -> n b t c")
        return x
