import torch
from torch import nn
from .activations import activations


class ConvDiscriminator(nn.Module):

    def __init__(self, audio_feature_size, dir_size, **kwargs):

        super().__init__()

        self.audio_feature_size = audio_feature_size
        self.dir_size = dir_size

        n_poses = kwargs['n_poses']
        hidden_size = kwargs['hidden_size']
        batchnorm = kwargs['batchnorm']
        layernorm = kwargs['layernorm']

        self.activation = activations[kwargs['activation']]

        self.audio_feature_extractor = nn.Sequential(
            nn.Linear(audio_feature_size, hidden_size//4),
            self.activation,
            nn.Linear(hidden_size//4, hidden_size//2)
        )

        self.pose_feature_extractor = nn.Sequential(
            nn.Linear(dir_size, hidden_size//4),
            self.activation,
            nn.Linear(hidden_size//4, hidden_size//2)
        )

        if layernorm:

            if n_poses == 34:
                self.conv_layers = nn.Sequential(
                    self.conv_block(hidden_size, hidden_size, downsample=False, batchnorm=False),
                    nn.LayerNorm((hidden_size, 32)),
                    self.conv_block(hidden_size, 2 * hidden_size, downsample=True, batchnorm=False),
                    nn.LayerNorm((2*hidden_size, 15)),

                    self.conv_block(2 * hidden_size, 2 * hidden_size, downsample=False, batchnorm=False),
                    nn.LayerNorm((2*hidden_size, 13)),
                    self.conv_block(2 * hidden_size, 4 * hidden_size, downsample=True, batchnorm=False),
                    nn.LayerNorm((4*hidden_size, 5)),

                    self.conv_block(4 * hidden_size, 4 * hidden_size, downsample=False, batchnorm=False),
                    nn.LayerNorm((4*hidden_size, 3)),

                    nn.Conv1d(4 * hidden_size, 8 * hidden_size, 3)
                ) # for 34 frames

        else:
            self.conv_layers = nn.Sequential(
                self.conv_block(hidden_size, hidden_size, downsample=False, batchnorm=batchnorm),
                self.conv_block(hidden_size, 2 * hidden_size, downsample=True, batchnorm=batchnorm),

                self.conv_block(2 * hidden_size, 2 * hidden_size, downsample=False, batchnorm=batchnorm),
                self.conv_block(2 * hidden_size, 4 * hidden_size, downsample=True, batchnorm=batchnorm),

                self.conv_block(4 * hidden_size, 4 * hidden_size, downsample=False, batchnorm=batchnorm),
                nn.Conv1d(4 * hidden_size, 8 * hidden_size, 3)
            ) # for 34 frames

        if batchnorm:
            self.out_net = nn.Sequential(
                nn.Linear(1024, 256), 
                nn.BatchNorm1d(256),
                self.activation,
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                self.activation,
                nn.Linear(64, 1),
            )
        elif layernorm:
            self.out_net = nn.Sequential(
                nn.Linear(8 * hidden_size, 256), 
                nn.LayerNorm(256),
                self.activation,
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                self.activation,
                nn.Linear(64, 1),
            )
        else:
            self.out_net = nn.Sequential(
                nn.Linear(1024, 256),
                self.activation,
                nn.Linear(256, 64),
                self.activation,
                nn.Linear(64, 1),
            )

    def conv_block(self, in_channels, out_channels, downsample=False, padding=0, batchnorm=False):

        if downsample:
            kernel_size = 4
            stride = 2
        else:
            kernel_size = 3
            stride = 1
            
        if batchnorm:
            net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm1d(out_channels),
                self.activation
            )

        else:
            net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
                self.activation
            )
        return net


    def forward(self, in_dir, in_audio, debug=False):

        x_audio = self.audio_feature_extractor(in_audio)
        x_pose = self.pose_feature_extractor(in_dir)
        x = torch.cat([x_audio, x_pose], dim=-1) # (N, T, hidden_size)

        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        if debug:
            print(x.shape)
            assert 0
        x = x.transpose(1, 2)
        x = x.view(x.size(0), -1)

        return self.out_net(x)

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


if __name__ == "__main__":

    T = 34

    in_dir = torch.Tensor(2, T, 27).normal_()
    in_audio = torch.Tensor(2, T, 2).normal_()
    disc = ImprovedConvDiscriminator(2, 27, d_model=128, activation='leaky_relu')

    # print(disc.count_parameters())

    disc(in_dir, in_audio)

    # print(disc)

    # gru = JitBiGRU(8, 256, num_layers=4, mode='sum', dropout=0.2)

    # x = torch.Tensor(1, 28, 8).normal_()

    # gru(x)
