import torch
import torch.nn.functional as F
from e2cnn import gspaces, nn
from torch.nn import Module


def get_non_linearity(scalar_fields, vector_fields):
    out_type = scalar_fields + vector_fields
    relu = nn.ReLU(scalar_fields)
    norm_relu = nn.NormNonLinearity(vector_fields)
    nonlinearity = nn.MultipleModule(
        out_type,
        ["relu"] * len(scalar_fields) + ["norm"] * len(vector_fields),
        [(relu, "relu"), (norm_relu, "norm")],
    )
    return nonlinearity


def get_batch_norm(scalar_fields, vector_fields):
    out_type = scalar_fields + vector_fields
    batch_norm = nn.InnerBatchNorm(scalar_fields)
    norm_batch_norm = nn.NormBatchNorm(vector_fields)
    batch_norm = nn.MultipleModule(
        out_type,
        ["bn"] * len(scalar_fields) + ["nbn"] * len(vector_fields),
        [(batch_norm, "bn"), (norm_batch_norm, "nbn")],
    )
    return batch_norm


class Encoder(Module):
    def __init__(self, out_dim, hidden_dim=32, pool=False, in_channel=1):
        super().__init__()
        self.out_dim = out_dim
        self.r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=8)

        in_type = nn.FieldType(self.r2_act, in_channel * [self.r2_act.trivial_repr])
        self.input_type = in_type
        self.pool = pool

        # convolution 1
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block1 = nn.SequentialModule(
            # nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            batch_norm,
            nonlinearity,
        )

        # convolution 2
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity,
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity,
        )

        # convolution 4
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity,
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity,
        )

        # convolution 6
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
            batch_norm,
            nonlinearity,
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        # convolution 7 --> out
        # the old output type is the input type to the next layer
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, out_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, 1 * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field

        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=1, padding=1, bias=False),
        )

        if self.pool is not True:
            self.layers = [
                self.block1,
                self.block2,
                self.block3,
                self.block4,
                self.block5,
                self.block6,
                self.block7,
            ]
        else:
            self.layers = [
                self.block1,
                self.block2,
                self.pool1,
                self.block3,
                self.block4,
                self.pool2,
                self.block5,
                self.block6,
                self.pool3,
                self.block7,
            ]

    def conv_forward(self, x: torch.Tensor, return_sizes=False):

        if return_sizes:
            sizes = [None] * len(self.layers)
            hidden_channels = [None] * len(self.layers)
        if len(x.shape) < 4:
            x = x.unsqueeze(1)

        x = nn.GeometricTensor(x, self.input_type)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_sizes:
                sizes[i] = x.shape[2:]
                hidden_channels[i] = x.shape[1]

        if return_sizes:
            return x, sizes, hidden_channels
        else:
            return x

    def forward(self, x: torch.Tensor):

        x = self.conv_forward(x, return_sizes=False)

        # x = x.tensor.squeeze(-1).squeeze(-1)
        x = x.tensor.mean(dim=(2, 3))

        x_0, x_1 = x[:, : self.out_dim], x[:, self.out_dim :]

        return x_0, x_1


class Decoder(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # convolution 1
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_size,
                hidden_size,
                kernel_size=1,
                padding=0,
            ),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
        )

        # convolution 2
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
        )

        # convolution 3
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
        )

        # convolution 4
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
        )

        # convolution 5
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
        )

        # convolution 6
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, 1, kernel_size=1, padding=0),
        )

        self.scale_factor = 2.24
        # self.scale_factor = 2

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1).unsqueeze(-1)  # [bz, emb_dim, 1, 1]
        x = x.expand(-1, -1, 2, 2)
        # pos_emb = torch.Tensor([[1, 2], [4, 3]]).type_as(x).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1, -1)
        # x = x + pos_emb

        x = self.block1(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=self.scale_factor)
        x = self.block2(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=self.scale_factor)
        x = self.block3(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=self.scale_factor)
        x = self.block4(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=self.scale_factor)
        x = self.block5(x)
        x = self.block6(x)
        # x = x[:, :, 2:30, 2:30]
        x = x[:, :, 3:35, 3:35]
        # x = torch.sigmoid(x)
        return x