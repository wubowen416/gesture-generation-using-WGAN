import torch
import torch.nn as nn


def linear_block(d_in, d_out):
    return  nn.Sequential(
            nn.Linear(d_in, d_out//2),
            nn.LayerNorm(d_out//2),
            nn.GELU(),
            nn.Linear(d_out//2, d_out),
            nn.LayerNorm(d_out),
            nn.GELU()
        )


class PoseGenerator(nn.Module):

    def __init__(self, d_cond, d_data, d_noise, d_model, num_layers=1, bidirectional=True):

        super().__init__()

        self.d_model = d_model
        self.d_noise = d_noise

        self.f_cond = linear_block(d_cond, d_model//3)
        self.f_noise = linear_block(d_noise, d_model//3)
        self.f_pre_data = linear_block(d_data, d_model//3)
        self.proj_agg = nn.Linear(3*(d_model//3), d_model)
        self.gru_layers = nn.ModuleList([nn.GRU(d_model, d_model, num_layers=1, batch_first=True, bidirectional=bidirectional) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.f_out = linear_block(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_data)

    def forward(self, cond, pre_data):
        """
        Args:
            cond - Tensor, shape [batch_size, time_step, d_cond]
            pre_data - Tensor, shape [batch_size, time_step, d_data]
        
        """
        noise = self.sample_noise(cond.size(0), cond.device)

        x_cond = self.f_cond(cond)
        x_pre_data = self.f_pre_data(pre_data)
        x_noise = self.f_noise(noise)
        x_noise = x_noise.unsqueeze(1).repeat(1, x_cond.size(1), 1)

        x = torch.cat([x_cond, x_noise, x_pre_data], dim=-1)
        x = self.proj_agg(x)

        for _, (rnn, norm) in enumerate(zip(self.gru_layers, self.norm_layers)):

            x_gru, _ = rnn(x)
            x_gru = x_gru[:, :, :self.d_model] + x_gru[:, :, self.d_model:]  # sum bidirectional outputs
            x = norm(x + x_gru) # Residual

        x = self.f_out(x)
        x = self.proj_out(x) + pre_data # Residual
        return x

    def sample_noise(self, batch_size, device, mean=0, std=1):
        return torch.normal(mean=mean, std=std, size=(batch_size, self.d_noise), device=device)

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


if __name__ == "__main__":

    pg = PoseGenerator(d_cond=2, d_noise=20, d_data=36, d_model=256, num_layers=2, bidirectional=True)

    pre_poses = torch.zeros((2, 34, 36))
    in_audio = torch.zeros((2, 34, 2))

    pg(in_audio, pre_poses)

    print(pg.count_parameters())
