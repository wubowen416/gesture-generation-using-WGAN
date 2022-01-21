import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def conv_block(d_in, d_out, downsample=False):

        if downsample:
            kernel_size = 4
            stride = 2
        else:
            kernel_size = 3
            stride = 1
            
        net = nn.Sequential(
            spectral_norm(nn.Conv1d(d_in, d_out, kernel_size, stride, padding=0)),
            nn.LayerNorm(d_out),
            nn.GELU()
        )
        return net
    


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class ConditionalConvDiscriminator(nn.Module):

    def __init__(self, d_data, d_cond, d_model):

        super().__init__()

        self.f_data = nn.Sequential(
            nn.Linear(d_data, d_model//4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(d_model//4, d_model//2)
        )

        self.f_cond = nn.Sequential(
            nn.Linear(d_cond, d_model//4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(d_model//4, d_model//2)
        )

        self.f_conv = nn.Sequential(
            conv_block(d_model, d_model, downsample=False, batchnorm=False),
            Permute([0, 2, 1]),
            nn.LayerNorm((128)),
            Permute([0, 2, 1]),
            conv_block(d_model, 2 * d_model, downsample=True, batchnorm=False),
            Permute([0, 2, 1]),
            nn.LayerNorm((256)),
            Permute([0, 2, 1]),

            conv_block(2 * d_model, 2 * d_model, downsample=False, batchnorm=False),
            Permute([0, 2, 1]),
            nn.LayerNorm((256)),
            Permute([0, 2, 1]),
            conv_block(2 * d_model, 4 * d_model, downsample=True, batchnorm=False),
            Permute([0, 2, 1]),
            nn.LayerNorm((512)),
            Permute([0, 2, 1]),

            conv_block(4 * d_model, 4 * d_model, downsample=False, batchnorm=False),
            Permute([0, 2, 1]),
            nn.LayerNorm((512)),
            Permute([0, 2, 1]),

            nn.Conv1d(4 * d_model, 8 * d_model, 3)
        ) # for 34 frames

        self.f_out = nn.Sequential(
            nn.Linear(8*d_model, 256), 
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
        )

    def forward(self, x_data, x_cond):
        """
        Args:
            x_data - Tensor, shape [batch_size, time_step, d_data]
            x_cond - Tensor, shape [batch_size, time_step, d_cond]
        Returns:
            Tensor, shape [batch_size, 1]
        """

        x_cond = self.f_cond(x_cond)
        x_data = self.f_data(x_data)
        x = torch.cat([x_cond, x_data], dim=-1) # (N, T, hidden_size)

        x = x.transpose(1, 2)
        x = self.f_conv(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), -1)
        return self.f_out(x)

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class ConvDiscriminator(nn.Module):

    def __init__(self, d_data, d_model):

        super().__init__()

        self.f_data = nn.Sequential(
            nn.Linear(d_data, d_model//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(d_model//2, d_model)
        )

        self.f_conv = nn.Sequential(
            conv_block(d_model, d_model, downsample=False, batchnorm=False),
            Permute([0, 2, 1]),
            nn.LayerNorm((128)),
            Permute([0, 2, 1]),
            conv_block(d_model, 2 * d_model, downsample=True, batchnorm=False),
            Permute([0, 2, 1]),
            nn.LayerNorm((256)),
            Permute([0, 2, 1]),

            conv_block(2 * d_model, 2 * d_model, downsample=False, batchnorm=False),
            Permute([0, 2, 1]),
            nn.LayerNorm((256)),
            Permute([0, 2, 1]),
            conv_block(2 * d_model, 4 * d_model, downsample=True, batchnorm=False),
            Permute([0, 2, 1]),
            nn.LayerNorm((512)),
            Permute([0, 2, 1]),

            conv_block(4 * d_model, 4 * d_model, downsample=False, batchnorm=False),
            Permute([0, 2, 1]),
            nn.LayerNorm((512)),
            Permute([0, 2, 1]),

            nn.Conv1d(4 * d_model, 8 * d_model, 3)
        ) # for 34 frames

        self.f_out = nn.Sequential(
            nn.Linear(8*d_model, 256), 
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Args:
            x_data - Tensor, shape [batch_size, time_step, d_data]
        Returns:
            Tensor, shape [batch_size, 1]
        """
        x = self.f_data(x)
        x = x.transpose(1, 2)
        x = self.f_conv(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), -1)
        return self.f_out(x)

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


if __name__ == "__main__":

    x_data = torch.Tensor(2, 34, 36).normal_()
    x_cond = torch.Tensor(2, 34, 2).normal_()
    disc = ConditionalConvDiscriminator(d_data=36, d_cond=2, d_model=128)

    disc(x_data, x_cond)

    disc2 = ConvDiscriminator(d_data=36, d_model=128)

    disc2(x_data)
