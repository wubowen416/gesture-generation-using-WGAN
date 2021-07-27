import torch
from torch import nn

from .jit_gru import JitGRU


class JitBiGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, bias=True, mode='sum', dropout=0.2):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.mode = mode
        self.dropout = dropout

        if num_layers == 1:
            self.forward_rnn_layers = nn.ModuleList(
                [JitGRU(input_size, hidden_size, num_layers=1, batch_first=batch_first, bias=bias)])
            self.backward_rnn_layers = nn.ModuleList(
                [JitGRU(input_size, hidden_size, num_layers=1, batch_first=batch_first, bias=bias)])

        else:
            self.forward_rnn_layers = nn.ModuleList([JitGRU(input_size, hidden_size, num_layers=1, batch_first=batch_first, bias=bias)] + [
                                                    JitGRU(hidden_size, hidden_size, num_layers=1, batch_first=batch_first, bias=bias) for _ in range(num_layers - 1)])
            self.backward_rnn_layers = nn.ModuleList([JitGRU(input_size, hidden_size, num_layers=1, batch_first=batch_first, bias=bias)] + [
                                                     JitGRU(hidden_size, hidden_size, num_layers=1, batch_first=batch_first, bias=bias) for _ in range(num_layers - 1)])

        if dropout:
            self.dropout_layer = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):

        for forward_layer, backward_layer in zip(self.forward_rnn_layers, self.backward_rnn_layers):

            x_forward, _ = forward_layer(x)
            x_backward, _ = backward_layer(torch.flip(x, dims=[1]))

            if self.mode == 'sum':
                x = x_forward + x_backward

            if self.dropout:
                x = self.dropout_layer(x)

        return x


class GRUDiscriminator(nn.Module):

    def __init__(self, audio_feature_size, dir_size, hidden_size, dropout=0.2):

        super().__init__()

        self.audio_feature_size = audio_feature_size
        self.dir_size = dir_size
        self.hidden_size = hidden_size

        self.pre_conv = nn.Sequential(
            nn.Conv1d(audio_feature_size + dir_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(8, 8, 3),
        )

        self.gru = JitBiGRU(8, hidden_size, num_layers=2,
                            mode='sum', dropout=0.2)
        self.out = nn.Linear(hidden_size, 1)
        self.out2 = nn.Linear(28, 1)

    def forward(self, in_dir, in_audio):

        x = torch.cat([in_dir, in_audio], dim=-1)  # (N, T, 29)

        x = x.transpose(1, 2)
        x = self.pre_conv(x)
        x = x.transpose(1, 2)

        x = self.gru(x)

        x = self.out(x)
        x = x.squeeze(-1)
        return self.out2(x)

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class SelfAttention(nn.Module):

    def __init__(self, in_size, h_size):

        super(SelfAttention, self).__init__()

        self.in_size = in_size
        self.h_size = h_size

        self.weight_q = nn.Parameter(torch.nn.init.normal_(
            torch.Tensor(in_size, h_size), 0.0, 0.02), requires_grad=True)
        self.weight_k = nn.Parameter(torch.nn.init.normal_(
            torch.Tensor(in_size, h_size), 0.0, 0.02), requires_grad=True)

    def forward(self, x):
        Q = x @ self.weight_q
        K = x @ self.weight_k
        logit = torch.einsum('tbd,tbd -> tb', Q, K).unsqueeze(-1) / np.sqrt(self.h_size) # (T, B, 1)
        att = torch.softmax(logit, dim=0)  # for each time step (T, B, 1)
        return att * x, att


class ConvDiscriminator(nn.Module):

    def __init__(self, audio_feature_size, dir_size, n_poses, hidden_size, batchnorm=False, layernorm=False, sa=False):

        super().__init__()

        self.audio_feature_size = audio_feature_size
        self.dir_size = dir_size
        self.sa = sa

        self.audio_feature_extractor = nn.Sequential(
            nn.Linear(audio_feature_size, hidden_size//4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//4, hidden_size//2)
        )

        self.pose_feature_extractor = nn.Sequential(
            nn.Linear(dir_size, hidden_size//4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//4, hidden_size//2)
        )

        if layernorm:

            self.conv_layers = nn.Sequential(
                self.conv_block(hidden_size, hidden_size, downsample=False, batchnorm=False),
                nn.LayerNorm((128, 32)),
                self.conv_block(hidden_size, 2 * hidden_size, downsample=True, batchnorm=False),
                nn.LayerNorm((256, 15)),

                self.conv_block(2 * hidden_size, 2 * hidden_size, downsample=False, batchnorm=False),
                nn.LayerNorm((256, 13)),
                self.conv_block(2 * hidden_size, 4 * hidden_size, downsample=True, batchnorm=False),
                nn.LayerNorm((512, 5)),

                self.conv_block(4 * hidden_size, 4 * hidden_size, downsample=False, batchnorm=False),
                nn.LayerNorm((512, 3)),

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
                nn.LeakyReLU(0.2, True),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2, True),
                nn.Linear(64, 1),
            )
        elif layernorm:
            self.out_net = nn.Sequential(
                nn.Linear(1024, 256), 
                nn.LayerNorm(256),
                nn.LeakyReLU(0.2, True),
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(0.2, True),
                nn.Linear(64, 1),
            )
        else:
            self.out_net = nn.Sequential(
                nn.Linear(1024, 256),
                nn.LeakyReLU(0.2, True),
                nn.Linear(256, 64),
                nn.LeakyReLU(0.2, True),
                nn.Linear(64, 1),
            )

        if sa:
            self.sa_layer = SelfAttention(hidden_size, hidden_size)

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
                nn.LeakyReLU(0.2, True)
            )

        else:
            net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
                nn.LeakyReLU(0.2, True)
            )
        return net


    def forward(self, in_dir, in_audio):

        x_audio = self.audio_feature_extractor(in_audio)
        x_pose = self.pose_feature_extractor(in_dir)
        x = torch.cat([x_audio, x_pose], dim=-1) # (N, T, hidden_size)

        if self.sa:
            x, att_mat = self.sa_layer(x)

        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), -1)

        return self.out_net(x)

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


if __name__ == "__main__":

    in_dir = torch.Tensor(2, 34, 27).normal_()
    in_audio = torch.Tensor(2, 34, 2).normal_()
    disc = ConvDiscriminator(2, 27, 34, 128, batchnorm=False, layernorm=True)

    print(disc.count_parameters())

    disc(in_dir, in_audio)

    # print(disc)

    # gru = JitBiGRU(8, 256, num_layers=4, mode='sum', dropout=0.2)

    # x = torch.Tensor(1, 28, 8).normal_()

    # gru(x)
