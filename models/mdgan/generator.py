import torch
import torch.nn as nn


class PoseGenerator(nn.Module):

    def __init__(self, d_cond, d_data, chunk_len, d_noise, d_model, num_layers=1, bidirectional=True, dropout=0):

        super().__init__()

        self.d_model = d_model
        self.d_noise = d_noise
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=dropout)

        self.f_cond = nn.Sequential(
            nn.Linear(d_cond, d_model//4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(d_model//4, d_model//2)
        )

        self.f_noise = nn.Sequential(
            nn.Linear(d_noise, d_model//4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(d_model//4, d_model//2)
        )

        self.f_pre_data = nn.Sequential(
            nn.Linear(d_data, d_model//4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(d_model//4, d_model//2)
        )

        self.gru = nn.ModuleList([nn.GRU(d_model+d_model//2, d_model, num_layers=1, batch_first=True, bidirectional=bidirectional)] + [
                                    nn.GRU(d_model, d_model, num_layers=1, batch_first=True, bidirectional=True) for _ in range(num_layers-1)])
        self.layernorm = nn.ModuleList([nn.LayerNorm((chunk_len, d_model)) for _ in range(num_layers)])

        self.f_out = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(d_model//2, d_data)
        )

    def forward(self, x_cond, x_pre_data):
        """
        Args:
            x_cond - Tensor, shape [batch_size, time_step, d_cond]
            x_pre_data - Tensor, shape [batch_size, time_step, d_data]
        
        """
        x_noise = self.sample_noise(x_cond.size(0), x_cond.device)

        x_cond = self.f_cond(x_cond)
        x_pre_data = self.f_pre_data(x_pre_data)
        x_noise = self.f_noise(x_noise)
        x_noise = x_noise.unsqueeze(1).repeat(1, x_cond.size(1), 1)

        x = torch.cat([x_cond, x_noise, x_pre_data], dim=-1)

        for idx_layer, (gru_layer, norm_layer) in enumerate(zip(self.gru, self.layernorm)):

            x, _ = gru_layer(x)
            x = x[:, :, :self.d_model] + x[:, :, self.d_model:]  # sum bidirectional outputs
            x = norm_layer(x)

            if idx_layer < self.num_layers:
                x = self.dropout(x)

        x = self.f_out(x)
        return x

    def sample_noise(self, batch_size, device, mean=0, std=1):
        return torch.normal(mean=mean, std=std, size=(batch_size, self.d_noise), device=device)

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


if __name__ == "__main__":

    pg = PoseGenerator(d_cond=2, d_noise=20, d_data=36, d_model=256, chunk_len=34, num_layers=1, bidirectional=True, dropout=0)

    pre_poses = torch.zeros((2, 34, 36))
    in_noise = torch.normal(mean=0, std=1, size=(2, 20))
    in_audio = torch.zeros((2, 34, 2))

    pg(in_noise, in_audio, pre_poses)

    print(pg.count_parameters())
