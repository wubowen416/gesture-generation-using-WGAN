import torch
import torch.nn as nn
import math
from typing import Tuple
from torch import Tensor

class AddNorm(nn.Module):
    def __init__(self, d_model: int, chunk_len: int, dropout: float = 0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm([chunk_len, d_model])

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x_0: Tensor, shape [batch_size, seq_len, d_model]
            x_1: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.layernorm(self.dropout(x + y))


class PositionWiseFFN(nn.Module):
    def __init__(self, units, hidden_size):
        super(PositionWiseFFN, self).__init__()
        self.ffn_1 = nn.Linear(hidden_size, activation='relu')
        self.ffn_2 = nn.Linear(units)

    def forward(self, X):
        return self.ffn_2(self.ffn_1(X))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, input: Tensor, start_position: int = 0) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model],
            or shape [batch_size, seq_len, d_model] if batch_first is True
        Returns:
            x: Tensor, shape [seq_len, batch_size, d_model],
            or shape [batch_size, seq_len, d_model] if batch_first is True
        """
        if self.batch_first:
            x = input.transpose(0, 1)
        x += self.pe[start_position:x.size(0) + start_position]
        if self.batch_first:
            x = x.transpose(0, 1)
        return x


def transpose_qkv(X, num_heads):
    # Shape after reshape: (batch_size, num_items, num_heads, p)
    # 0 means copying the shape element, -1 means inferring its value
    X = X.view((0, 0, num_heads, -1))
    # Swap the num_items and the num_heads dimensions
    X = X.permute((0, 2, 1, 3))
    # Merge the first two dimensions. Use reverse=True to infer
    # shape from right to left
    return X.view((-1, 0, 0), reverse=True)


def transpose_output(X, num_heads):
    # A reversed version of transpose_qkv
    X = X.reshape((-1, num_heads, 0, 0), reverse=True)
    X = X.transpose((0, 2, 1, 3))
    return X.reshape((0, 0, -1))


class MultiHeadAttention(nn.Block):
    def __init__(self, units, num_heads, dropout, **kwargs):  # units = d_o
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert units % num_heads == 0
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(units, use_bias=False, flatten=False)
        self.W_k = nn.Dense(units, use_bias=False, flatten=False)
        self.W_v = nn.Dense(units, use_bias=False, flatten=False)

    # query, key, and value shape: (batch_size, num_items, dim)
    # valid_length shape is either (bathc_size, ) or (batch_size, num_items)
    def forward(self, query, key, value, valid_length):
        # Project and transpose from (batch_size, num_items, units) to
        # (batch_size * num_heads, num_items, p), where units = p * num_heads.
        query, key, value = [transpose_qkv(X, self.num_heads) for X in (
            self.W_q(query), self.W_k(key), self.W_v(value))]
        if valid_length is not None:
            # Copy valid_length by num_heads times
            if valid_length.ndim == 1:
                valid_length = valid_length.tile(self.num_heads)
            else:
                valid_length = valid_length.tile((self.num_heads, 1))
        output = self.attention(query, key, value, valid_length)
        # Transpose from (batch_size * num_heads, num_items, p) back to
        # (batch_size, num_items, units)
        return transpose_output(output, self.num_heads)


class PoseGeneratorTransformer(nn.Module):
    def __init__(self, d_cond, d_noise, d_motion, chunk_len, d_model, dropout=0.1, ):
        super(PoseGeneratorTransformer, self).__init__()

        self.f_cond = nn.Linear(d_cond, d_model)
        self.f_noise = nn.Linear(d_noise, d_model)
        self.f_motion = nn.Linear(d_motion, d_model)

        d_model = 3 * d_model
        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout, batch_first=True)


    def forward(self, input_prev_motion, input_noise, input_cond):
        T = input_cond.shape[1]
        x_prev_motion = self.f_motion(input_prev_motion)
        x_noise = self.f_noise(input_noise)
        x_noise = x_noise.unsqueeze(1).repeat(1, T, 1)
        x_cond = self.f_cond(input_cond)
        x = torch.cat([x_prev_motion, x_noise, x_cond], dim=-1)
        print(x.shape)


        
    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


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


if __name__ == "__main__":

    # pg = PoseGenerator(2, 20, 27, 34, 256, num_layers=2, dropout=0.2, layernorm=True)
    pg = PoseGeneratorTransformer(2, 20, 27, 34, 256, dropout=0.2)

    pre_poses = torch.Tensor(2, 34, 27).normal_()
    in_noise = torch.normal(mean=0, std=1, size=(2, 20))
    in_audio = torch.Tensor(2, 34, 2).normal_()

    pg(pre_poses, in_noise, in_audio)

    print(pg.count_parameters())
