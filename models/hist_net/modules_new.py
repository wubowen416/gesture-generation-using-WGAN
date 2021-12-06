from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch import Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, 1)
        pe[:, 0::2, 0] = torch.sin(position * div_term)
        pe[:, 1::2, 0] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.batch_first = batch_first

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, d_model, batch_size, ?]
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        pe = self.pe[:x.size(0)]
        # Deal with dimensional more than 3
        if len(x.shape) > 3:
            for _ in range(len(x.shape) - 3):
                pe = pe.squeeze(-1)
        x += pe

        if self.batch_first:
            x = x.transpose(0, 1)

        return self.dropout(x)


class HistNet(nn.Module):

    def __init__(self, d_cond: int, d_motion: int, d_output: int, d_model: int = 128, dropout: int = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        self.f_emb = Embedding(d_cond=d_cond, d_motion=d_motion, d_model=d_model, dropout=dropout)
        # Align inputs
        self.f_inputs = InputAligner(d_model=d_model, dropout=dropout)

    def forward(self, cond_chunks: Tensor, prev_motion_chunks: Tensor):
        """
        Args:
            cond_chunks - Tensor, shape (batch_size, num_chunk, chunk_len, d_cond)
            prev_motion_chunks - Tensor, shape (batch_size, num_chunk, chunk_len, d_motion)
        """
        B, N, T, _ = cond_chunks.shape

        x_cond_chunks, x_prev_motion_chunks = self.f_emb(cond_chunks=cond_chunks, prev_motion_chunks=prev_motion_chunks)

        for n_step in range(N):

            self._step(x_cond=x_cond_chunks[:, n_step], x_prev_motion=x_prev_motion_chunks[:, n_step])

    
    def _step(self, x_cond: Tensor, x_prev_motion: Tensor):
        """
        Args:
            cond - Tensor, shape (batch_size, chunk_len, d_cond)
            prev_motion - Tensor, shape (batch_size, chunk_len, d_motion)
        """
        self.f_inputs(x_cond=x_cond, x_prev_motion=x_prev_motion)



class InputAligner(nn.Module):

    def __init__(self, d_model: int = 128, num_layers=1, bidirectional=True, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        

    def forward(self, x_cond: Tensor, x_prev_motion: Tensor) -> Tuple():
        """
        Args:
            x_cond: Tensor, shape [batch_size, chunk_len, d_model]
            x_prev_motion: Tensor, shape [batch_size, chunk_len, d_model]
        """

        


if __name__ == '__main__':

    D_CHUNK = 34
    D_COND = 2
    D_MOTION = 36
    BATCH_SIZE = 5

    cond_chunks = torch.zeros(BATCH_SIZE, 20, D_CHUNK, D_COND)
    prev_motion_chunks = torch.zeros(BATCH_SIZE, 20, D_CHUNK, D_MOTION)

    net = HistNet(d_cond=D_COND, d_motion=D_MOTION, d_output=D_MOTION)

    net(cond_chunks=cond_chunks, prev_motion_chunks=prev_motion_chunks)