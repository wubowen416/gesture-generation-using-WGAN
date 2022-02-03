from torch import nn

activations = dict(
    gelu = nn.GELU(),
    leaky_relu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
)