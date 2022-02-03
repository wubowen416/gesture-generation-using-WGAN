import torch
import torch.nn as nn
from .activations import activations


class PoseGenerator(nn.Module):
    def __init__(self, audio_feature_size, dir_size, **kwargs):
        super().__init__()

        self.hidden_size = kwargs['hidden_size']
        self.audio_feature_size = audio_feature_size
        self.noise_size = kwargs['noise_dim']
        self.dir_size = dir_size
        self.layernorm = kwargs['layernorm']
        self.n_poses = kwargs['n_poses']
        self.num_layers = kwargs['num_layers']

        self.activation = activations[kwargs['activation']]

        self.audio_feature_extractor = nn.Sequential(
            nn.Linear(audio_feature_size, self.hidden_size//4),
            self.activation,
            nn.Linear(self.hidden_size//4, self.hidden_size//2)
        )

        self.noise_processor = nn.Sequential(
            nn.Linear(self.noise_size, self.hidden_size//4),
            self.activation,
            nn.Linear(self.hidden_size//4, self.hidden_size//2)
        )

        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            self.activation,
            nn.Linear(self.hidden_size//2, dir_size)
        )

        if self.layernorm:
            self.gru = nn.ModuleList([nn.GRU(self.hidden_size + dir_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)] + [
                                     nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True) for _ in range(self.num_layers - 1)])
            self.layernorm = nn.ModuleList([nn.LayerNorm((self.n_poses, self.hidden_size)) for _ in range(self.num_layers)])
            self.dropout_layer = nn.Dropout(p=kwargs['dropout'])
        else:
            self.gru = nn.GRU(self.hidden_size + dir_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)

    def forward(self, pre_poses, in_audio):

        in_noise = self.sample_noise(pre_poses.size(0), device=pre_poses.device)

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

    def sample_noise(self, batch_size, device, mean=0, std=1):
        return torch.normal(mean=mean, std=std, size=(batch_size, self.noise_size), device=device)


class ImprovedPoseGenerator(nn.Module):
    def __init__(self, audio_feature_size, dir_size, **kwargs):
        super().__init__()

        self.hidden_size = kwargs['hidden']
        self.audio_feature_size = audio_feature_size
        self.noise_size = kwargs['noise_dim']
        self.dir_size = dir_size
        self.layernorm = kwargs['layernorm']
        self.n_poses = kwargs['n_poses']
        self.num_layers = kwargs['num_layers']

        self.activation = activations[kwargs['activation']]

        self.audio_feature_extractor = nn.Sequential(
            nn.Linear(audio_feature_size, self.hidden_size//4),
            self.activation,
            nn.Linear(self.hidden_size//4, self.hidden_size//2)
        )

        self.noise_processor = nn.Sequential(
            nn.Linear(self.noise_size, self.hidden_size//4),
            self.activation,
            nn.Linear(self.hidden_size//4, self.hidden_size//2)
        )

        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            self.activation,
            nn.Linear(self.hidden_size//2, dir_size)
        )

        if self.layernorm:
            self.gru = nn.ModuleList([nn.GRU(self.hidden_size + dir_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)] + [
                                     nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True) for _ in range(self.num_layers - 1)])
            self.layernorm = nn.ModuleList([nn.LayerNorm((self.n_poses, self.hidden_size)) for _ in range(self.num_layers)])
            self.dropout_layer = nn.Dropout(p=kwargs['dropout'])
        else:
            self.gru = nn.GRU(self.hidden_size + dir_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)

    def forward(self, pre_poses, in_audio):

        in_noise = self.sample_noise(pre_poses.size(0), device=pre_poses.device)

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
        return x.reshape(-1, T, self.dir_size) + pre_poses

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def sample_noise(self, batch_size, device, mean=0, std=1):
        return torch.normal(mean=mean, std=std, size=(batch_size, self.noise_size), device=device)


if __name__ == "__main__":

    pg = PoseGenerator(2, 20, 27, 34, 256, num_layers=2, dropout=0.2, layernorm=True)

    pre_poses = torch.Tensor(2, 34, 27).normal_()
    in_noise = torch.normal(mean=0, std=1, size=(2, 20))
    in_audio = torch.Tensor(2, 34, 2).normal_()

    pg(pre_poses, in_noise, in_audio)

    print(pg.count_parameters())
