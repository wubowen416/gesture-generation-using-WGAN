import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Tuple
from torch import Tensor


def get_activation(name):
    if name == 'relu':
        return nn.ReLU()

def get_memory_cell(name):
    if name == 'lstm':
        return MemoryModuleLSTM

def get_pose_generator(name):
    if name == 'gru':
        return PoseGeneratorGRU

def get_chunk_connector(name):
    if name == 'interpolation':
        return ChunkConnectorInterpolation


class AddNorm(nn.Module):
    def __init__(self, d_model: int, chunk_len: int, dropout: float = 0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm([chunk_len, d_model])

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            y: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.layernorm(self.dropout(x + y))


class SequenceWiseFFN(nn.Module):
    def __init__(self, d_in: int, d_out: int, chunk_len: int, activation: str = 'relu', dropout: float = 0.1):
        super(SequenceWiseFFN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.f_0 = nn.Linear(in_features=d_in, out_features=d_out)
        self.f_1 = nn.Linear(in_features=chunk_len, out_features=chunk_len)
        if d_in == d_out:
            self.addnorm =True
            self.module_addnorm = AddNorm(d_model=d_out, chunk_len=chunk_len, dropout=dropout)
        self.activation = get_activation(name=activation)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: Tensor, shape [batch_size, seq_len, d_in]
        Returns:
            x: Tensor, shape [batch_size, seq_len, d_out]
        """
        x = self.activation(self.f_0(self.dropout(input)))
        x = self.f_1(self.dropout(x.transpose(1, 2))).transpose(1, 2)
        if self.addnorm:
            x = self.module_addnorm(x, input)
        return x


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
            input: Tensor, shape [seq_len, batch_size, d_model],
                or shape [batch_size, seq_len, d_model] if batch_first is True
            start_position: delay on time step for input
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


class Aligner(nn.Module):

    def __init__(self, d_cond: int, d_motion: int, d_model: int, chunk_len: int, prev_len: int, params: dict, activation: str = 'relu', dropout: float = 0.1):
        super(Aligner, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.f_cond = nn.Linear(in_features=d_cond, out_features=d_model//2)
        self.f_motion = nn.Linear(in_features=d_motion, out_features=d_model//2)
        if params['pe']:
            self.module_pe = PositionalEncoding(d_model=d_model//2, dropout=dropout, batch_first=True)
        self.f_fc = nn.Linear(d_model, d_model)
        self.module_addnorm = AddNorm(d_model=d_model, chunk_len=chunk_len, dropout=dropout)
        self.activation = get_activation(name=activation)

        self.chunk_len = chunk_len
        self.prev_len = prev_len
        self.pe = params['pe']
    
    def forward(self, input_cond: Tensor, input_prev_motion: Tensor) -> Tensor:
        """
        Args:
            input_cond: Tensor, shape [batch_size, seq_len, d_cond]
            input_prev_motion: Tensor, shape [batch_size, seq_len, d_motion]
        Returns:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x_cond = self.activation(self.f_cond(self.dropout(input_cond)))
        x_motion = self.activation(self.f_motion(self.dropout(input_prev_motion)))
        if self.pe:
            # Positional encoding with different position
            x_cond = self.module_pe(x_cond, start_position=self.chunk_len-self.prev_len)
            x_motion = self.module_pe(x_motion, start_position=0)
        x = torch.cat([x_cond, x_motion], dim=2)
        x = self.module_addnorm(self.f_fc(self.dropout(x)), x)
        return x


class MemoryModuleLSTM(nn.Module):

    def __init__(self, d_in: int, d_out: int, d_model: int, chunk_len: int, params: dict, activation: str = 'relu', dropout: float = 0.1):
        super(MemoryModuleLSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.module_swff = SequenceWiseFFN(d_in=d_in, d_out=d_model, chunk_len=chunk_len, activation=activation, dropout=dropout)
        self.f_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=params['num_layers'], bidirectional=False, dropout=dropout, batch_first=True)
        self.f_out = nn.Linear(in_features=d_model, out_features=d_out)
        self.module_addnorm = AddNorm(d_model=d_model, chunk_len=chunk_len, dropout=dropout)
        self.activation = get_activation(name=activation)
        self.d_model = d_model
        self.num_layers = params['num_layers']
    
    def activate_memory(self, batch_size: int):
        self.memory = (torch.zeros(self.num_layers, batch_size, self.d_model), torch.zeros(self.num_layers, batch_size, self.d_model))

    def asign_memory(self, memory: Tuple[Tensor, Tensor]):
        self.memory = memory

    def get_memory(self):
        return self.memory

    def check_memory_type(self, memory):
        assert type(memory) is Tuple, f"Must be (Tensor[num_layers, batch_size, d_model], Tensor[num_layers, batch_size, d_model]). Check memory type!"
        for i, m in enumerate(memory):
            assert len(m.shape) == 3, f"Element must be Tensor[num_layers, batch_size, d_model], Got {m.shape} at position {i}. Check memory type!"

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: Tensor, shape [batch_size, seq_len, d_in]
        Returns:
            x: Tensor, shape [batch_size, seq_len, d_out]
        """
        x = self.activation(self.module_swff(self.dropout(input)))
        x, self.memory = self.f_lstm(self.dropout(x), self.memory)
        x = self.f_out(self.dropout(x))
        x = self.module_addnorm(x, input)
        return x


class PoseGeneratorGRU(nn.Module):

    def __init__(self, d_in: int, d_out: int, d_model: int, chunk_len, params: dict, activation: str = 'relu', dropout: float = 0.1):
        super(PoseGeneratorGRU, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.f_swff = SequenceWiseFFN(d_in=d_in, d_out=d_model, chunk_len=chunk_len, activation=activation, dropout=dropout)
        self.f_gru = nn.GRU(d_model, d_model, num_layers=params['num_layers'], bidirectional=params['bidirectional'], dropout=dropout, batch_first=True)
        self.module_addnorm = AddNorm(d_model=d_model, chunk_len=chunk_len, dropout=dropout)
        self.f_out = nn.Linear(d_model, d_out)
        self.activation = get_activation(name=activation)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: Tensor, shape [batch_size, seq_len, d_in]
        Returns:
            x: Tensor, shape [batch_size, seq_len, d_out]
        """
        x = self.activation(self.f_swff(self.dropout(input)))
        x, _ = self.f_gru(self.dropout(x))
        x = x[:, :, :x.shape[-1]//2] + x[:, :, x.shape[-1]//2:]  # sum bidirectional outputs
        x = self.module_addnorm(x, input)
        x = self.f_out(self.dropout(x))
        return x


class ChunkConnectorInterpolation:

    def __init__(self, chunk_len: int, prev_len: int):
        self.chunk_len = chunk_len
        self.prev_len = prev_len

    def __call__(self, input: Tensor) -> Tensor:
        """
        Args:
            input: Tensor, shape [batch_size, seq_len, chunk_len, ?]
        Returns:
            output: Tensor, shape [batch_size, time_len, ?]
        """
        output = input[:, 0][:self.chunk_len-self.prev_len]
        n_chunk = input.shape[1]
        for i in range(1, n_chunk):
            trans_prev = input[:, i-1][:, -self.prev_len:]
            trans_next = input[:, i][:, :self.prev_len]
            # Transition strategy
            trans_motion = []
            for k in range(self.prev_len):
                trans = ((self.prev_len - k) / (self.prev_len + 1)) * trans_prev[:, k] + ((k + 1) / (self.prev_len + 1)) * trans_next[:, k]
                trans_motion.append(trans.unsqueeze(1))
            trans_motion = torch.cat(trans_motion, dim=1)
            # Append each
            if i != n_chunk - 1: # not last chunk
                output = torch.cat([output, trans_motion, input[:, i][:, self.prev_len:self.chunk_len-self.prev_len]], dim=1)
            else: # last chunk
                output = torch.cat([output, trans_motion, input[:, i][:, self.prev_len:self.chunk_len]], dim=1)
        return output


class HistNet(nn.Module):

    def __init__(self, 
                d_cond: int, 
                d_motion: int, 
                d_model: int, 
                chunk_len: int, 
                prev_len: int, 
                aligner_params: dict,
                memory_params: dict,
                pose_generator_parmas: dict,
                chunk_connector_params: dict,
                activation: str = 'relu', 
                dropout: float = 0.1):
        super(HistNet, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.module_align = Aligner(d_cond=d_cond, d_motion=d_motion, d_model=d_model, chunk_len=chunk_len, prev_len=prev_len, params=aligner_params, activation=activation, dropout=dropout)
        self.module_memory = get_memory_cell(memory_params['name'])(d_in=d_model, d_out=d_model, d_model=d_model, chunk_len=chunk_len, params=memory_params, activation='relu', dropout=0.1, )
        self.module_pose_generator = get_pose_generator(pose_generator_parmas['name'])(d_in=d_model, d_out=d_motion, d_model=d_model, chunk_len=chunk_len, params=pose_generator_parmas, activation=activation, dropout=dropout)
        self.module_chunk_connector = get_chunk_connector(chunk_connector_params['name'])(chunk_len=chunk_len, prev_len=prev_len)

    def init_memory(self, batch_size: int, memory = None):
        if memory is None:
            self.module_memory.activate_memory(batch_size)
        else:
            self.module_memory.check_memory_type(memory)
            self.module_memory.asign_memory(memory)

    def forward(self, cond_chunks: Tensor, prev_motion_chunks: Tensor, memory = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            cond_chunks: Tensor, shape [batch_size, seq_len, chunk_len, d_cond]
            prev_motion_chunks: Tensor, shape [batch_size, seq_len, chunk_len, d_motion]
        """
        self.init_memory(cond_chunks.shape[0], memory)

        x_chunks = list()
        for step in range(cond_chunks.shape[1]):

            cond = cond_chunks[:, step]
            prev_motion = prev_motion_chunks[:, step]

            x = self.module_align(cond, prev_motion)
            x = self.module_memory(x)
            x = self.module_pose_generator(x)

            x_chunks.append(x.unsqueeze(1))

        x_chunks = torch.cat(x_chunks, dim=1)
        x_seq = self.module_chunk_connector(x_chunks)
        
        return x_chunks, x_seq, self.module_memory.get_memory()


if __name__ == '__main__':

    MAX_TIME_STEP = 50
    CHUNK_LEN = 34
    PREV_LEN = 4
    D_COND = 2
    D_MOTION = 36
    D_MODEL = 128
    ACTIVATION = 'relu'
    DROPOUT = 0.1
    BATCH_SIZE = 5

    MEMORY_PARAMS = dict(
        name = 'lstm',
        num_layers = 2
    )

    ALIGHNER_PARMAS = dict(
        pe = True
    )

    POSE_GENERATOR_PARAMS = dict(
        name = 'gru',
        num_layers = 2,
        bidirectional = True
    )

    CHUNK_CONNECTOR_PARAMS = dict(
        name = 'interpolation'
    )

    cond_chunks = torch.zeros(BATCH_SIZE, MAX_TIME_STEP, CHUNK_LEN, D_COND)
    prev_motion_chunks = torch.zeros(BATCH_SIZE, MAX_TIME_STEP, CHUNK_LEN, D_MOTION)

    hist_net = HistNet(
        d_cond=D_COND, d_motion=D_MOTION, d_model=D_MODEL, 
        chunk_len=CHUNK_LEN, prev_len=PREV_LEN, aligner_params=ALIGHNER_PARMAS, 
        memory_params=MEMORY_PARAMS, pose_generator_parmas=POSE_GENERATOR_PARAMS, chunk_connector_params=CHUNK_CONNECTOR_PARAMS,
        activation=ACTIVATION, dropout=DROPOUT)

    hist_net(cond_chunks, prev_motion_chunks)