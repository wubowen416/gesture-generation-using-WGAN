import torch
import torch.nn as nn


class PoseGenerator(nn.Module):

    def __init__(self, audio_feature_size, noise_size, dir_size, n_poses, hidden_size, num_layers=2, dropout=0, layernorm=False):

        super().__init__()

        self.hidden_size = hidden_size
        self.audio_feature_size = audio_feature_size
        self.noise_size = noise_size
        self.dir_size = dir_size
        self.hidden_size = hidden_size
        self.layernorm = layernorm
        self.n_poses = n_poses
        self.num_layers = num_layers

        self.audio_feature_extractor = nn.Sequential(
            nn.Linear(audio_feature_size, hidden_size//4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//4, hidden_size//2)
        )

        self.noise_processor = nn.Sequential(
            nn.Linear(noise_size, hidden_size//4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//4, hidden_size//2)
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//2, dir_size)
        )

        if layernorm:
            self.gru = nn.ModuleList([nn.GRU(hidden_size + dir_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)] + [
                                     nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True) for _ in range(num_layers - 1)])
            self.layernorm = nn.ModuleList([nn.LayerNorm((n_poses, hidden_size)) for _ in range(num_layers)])
            self.dropout_layer = nn.Dropout(p=dropout, inplace=True)
        else:
            self.gru = nn.GRU(hidden_size + dir_size, hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, pre_poses, in_noise, in_audio):

        x_audio = self.audio_feature_extractor(in_audio)
        x_noise = self.noise_processor(in_noise)

        T = x_audio.size(1)
        x_noise = x_noise.unsqueeze(1).repeat(1, T, 1)

        x = torch.cat([x_audio, x_noise, pre_poses], dim=-1)

        if self.layernorm:

            for idx_layer, (gru_layer, norm_layer) in enumerate(zip(self.gru, self.layernorm)):

                x, _ = gru_layer(x)
                x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:]  # sum bidirectional outputs
                x = norm_layer(x)

                if idx_layer < self.num_layers:
                    x = self.dropout_layer(x)

        else:
            x, _ = self.gru(x)
            x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:]  # sum bidirectional outputs

        x = self.out(x.reshape(-1, x.shape[2]))
        return x.reshape(-1, T, self.dir_size)

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


# class PoseGenerator(nn.Module):

#     def __init__(self, d_cond, d_noise, d_pose, n_poses, d_model, num_layers=2, dropout=0):

#         super().__init__()

#         assert d_model % 3 == 0, "d_model must be divisible for 3."

#         self.d_model = d_model
#         self.d_cond = d_cond
#         self.d_noise = d_noise
#         self.d_pose = d_pose
#         self.d_model = d_model
#         self.n_poses = n_poses
#         self.num_layers = num_layers

#         self.proj_audio = nn.Linear(d_cond, d_model//3)
#         self.proj_noise = nn.Linear(d_noise, d_model//3)
#         self.proj_prepose = nn.Linear(d_pose, d_model//3)

#         self.f_cond = nn.Sequential(
#             nn.Linear(d_model//3, d_model//3),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(d_model//3, d_model//3),
#             nn.LeakyReLU(0.2, True)
#         )

#         self.f_noise = nn.Sequential(
#             nn.Linear(d_model//3, d_model//3),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(d_model//3, d_model//3),
#             nn.LeakyReLU(0.2, True)
#         )

#         self.f_prepose = nn.Sequential(
#             nn.Linear(d_model//3, d_model//3),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(d_model//3, d_model//3),
#             nn.LeakyReLU(0.2, True)
#         )

#         self.f_out = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(d_model, d_model),
#             nn.LeakyReLU(0.2, True)
#         )

#         self.proj_out = nn.Linear(d_model, d_pose)

#         self.rnn_layers = nn.ModuleList([nn.GRU(d_model, d_model, num_layers=1, batch_first=True, bidirectional=True) for _ in range(num_layers)])
#         self.norm_layers = nn.ModuleList([nn.LayerNorm((n_poses, d_model)) for _ in range(num_layers)])

#         self.dropout = nn.Dropout(p=dropout, inplace=True)

#     def forward(self, prepose, in_noise, in_audio):

#         x_audio = self.proj_audio(in_audio)
#         x_noise = self.proj_noise(in_noise)
#         x_prepose = self.proj_prepose(prepose)

#         x_audio = self.f_cond(x_audio) + x_audio
#         x_noise = self.f_noise(x_noise) + x_noise
#         x_prepose = self.f_prepose(x_prepose) + x_prepose

#         T = x_audio.size(1)
#         x_noise = x_noise.unsqueeze(1).repeat(1, T, 1)

#         x = torch.cat([x_audio, x_noise, x_prepose], dim=-1)

#         for idx_layer, (rnn, norm) in enumerate(zip(self.rnn_layers, self.norm_layers)):
#             x_rnn, _ = rnn(x)
#             x_rnn = x_rnn[:, :, :self.d_model] + x_rnn[:, :, self.d_model:]  # sum bidirectional outputs
#             x = norm(x + x_rnn)
#             x = self.dropout(x)

#         x = self.f_out(x) + x
#         x = self.proj_out(x) + prepose
#         return x

#     def count_parameters(self):
#         return sum([p.numel() for p in self.parameters() if p.requires_grad])


if __name__ == "__main__":

    pg = PoseGenerator(2, 20, 27, 34, 256, num_layers=2, dropout=0.2, layernorm=True)

    pre_poses = torch.Tensor(2, 34, 27).normal_()
    in_noise = torch.normal(mean=0, std=1, size=(2, 20))
    in_audio = torch.Tensor(2, 34, 2).normal_()

    pg(pre_poses, in_noise, in_audio)

    print(pg.count_parameters())
