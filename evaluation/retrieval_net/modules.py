import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import math
from typing import Tuple


def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky':
        return nn.LeakyReLU()
    elif name == 'tanh':
        return nn.Tanh()


class Embedder(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super(Embedder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed = nn.Linear(d_in, d_out)
    def forward(self, inputs: Tensor) -> Tensor:
        return self.embed(inputs)


class AddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_model)

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
        self.activation = get_activation(name=activation)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            input: Tensor, shape [batch_size, seq_len, d_in]
        Returns:
            x: Tensor, shape [batch_size, seq_len, d_out]
        """
        x = self.activation(self.f_0(self.dropout(inputs)))
        x = self.f_1(self.dropout(x.transpose(1, 2))).transpose(1, 2)
        return x


class PositionWiseFFN(nn.Module):
    def __init__(self, d_in: int, d_model: int, d_out: int, activation: str = 'relu', dropout: float = 0.1):
        super(PositionWiseFFN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.f_0 = nn.Linear(in_features=d_in, out_features=d_model)
        self.f_1 = nn.Linear(in_features=d_model, out_features=d_out)
        self.activation = get_activation(name=activation)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            input: Tensor, shape [batch_size, seq_len, d_in]
        Returns:
            x: Tensor, shape [batch_size, seq_len, d_out]
        """
        x = self.activation(self.f_0(self.dropout(inputs)))
        x = self.f_1(self.dropout(x))
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, inputs: Tensor, start_position: int = 0) -> Tensor:
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
            x = inputs.transpose(0, 1)
        x += self.pe[start_position:x.size(0) + start_position]
        if self.batch_first:
            x = x.transpose(0, 1)
        return x


class TransformerAttention(nn.Module):
    def __init__(self, d_in: int, d_model: int, d_out: int, num_heads: int = 1, mask_value: float = 1e-9, dropout: float = 0.1):
        super(TransformerAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.mask_value = mask_value
        
        self.f_q = nn.Linear(d_in, d_model)
        self.f_k = nn.Linear(d_in, d_model)
        self.f_v = nn.Linear(d_in, d_model)
        
        self.f_out = nn.Linear(d_model, d_out)
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q: Tensor, shape [B, T, d_in]
            k: Tensor, shape [B, T, d_in]
            v: Tensor, shape [B, T, d_in]
            mask: Tensor, shape [T, T]
            Note q == k == v
        Returns:
            output: Tensor, shape [B, T, d_out], weighted output
            scores: Tensor, shape [B, num_heads, T, T], attention scores
        """
        
        B, T, _ = q.shape
        
        # perform linear operation and split into h heads
        q = self.f_q(self.dropout(q)).view(B, T, self.num_heads, self.d_head)
        k = self.f_k(self.dropout(k)).view(B, T, self.num_heads, self.d_head)
        v = self.f_v(self.dropout(v)).view(B, T, self.num_heads, self.d_head)
        
        # transpose to get dimensions B * num_heads * seq_len * d_head
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        x, scores = self.transformer_attention(q, k, v, mask)
        
        # concatenate heads and put through final linear layer
        x = x.transpose(1,2).contiguous().view(B, T, self.d_model)
        output = self.f_out(self.dropout(x))
    
        return output, scores

    def transformer_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_head)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, self.mask_value)
        scores = F.softmax(scores, dim=-1)

        # scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output, scores


class SelfAttentiveLayer(nn.Module):

    def __init__(self, d_in: int, d_model: int, d_out: int, num_heads: int = 1, dropout: float = 0.1) -> None:
        super(SelfAttentiveLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.f_v = nn.Linear(d_in, d_model)
        self.f_s1 = nn.Linear(d_in, d_model)
        self.f_s2 = nn.Linear(d_model, num_heads)
        self.f_out = nn.Linear(d_model, d_out)
        
    
    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: Tensor, shape [B, T, d_in]
        Returns:
            outputs: Tensor, shape [B, T, d_out], weighted output
            scores: Tensor, shape [B, num_heads, T, T], attention scores
        """

        B, T, _ = inputs.shape
        
        v = self.f_v(self.dropout(inputs)).view(B, T, self.d_head, self.num_heads)
        v = v.permute(0, 3, 1, 2)

        # Compute attentive scores
        x = self.f_s1(self.dropout(inputs))
        x = torch.tanh(x)
        x = self.f_s2(self.dropout(inputs))
        x = x.transpose(1, 2)
        scores = F.softmax(x, dim=-1)

        # Weighted average, concat head
        x = torch.matmul(scores.unsqueeze(2), v).view(B, self.d_model)
        outputs = self.f_out(self.dropout(x))
        return outputs, scores


class TransformerEncoder(nn.Module):

    def __init__(
        self, 
        d_model: int,
        num_heads: int = 1, 
        mask_value: float = 1e-9, 
        activation: str = 'relu', 
        dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.module_transformer_attetion = TransformerAttention(
            d_in=d_model, d_model=d_model, d_out=d_model, num_heads=num_heads, mask_value=mask_value, dropout=dropout)
        self.module_addnorm_0 = AddNorm(d_model=d_model, dropout=dropout)
        self.module_pwffn = PositionWiseFFN(d_in=d_model, d_model=d_model, d_out=d_model, activation=activation, dropout=dropout)
        self.module_addnorm_1 = AddNorm(d_model=d_model, dropout=dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: Tensor, shape [N, T, d_model]
        Returns:
            outputs: Tensor, shape [N, T, d_model]
        """
        x_1, _ = self.module_transformer_attetion(inputs, inputs, inputs, mask=None)
        x = self.module_addnorm_0(inputs, x_1)
        x_1 = self.module_pwffn(x)
        outputs = self.module_addnorm_1(x, x_1)
        return outputs


class AttentiveEncoder(nn.Module):

    def __init__(self, 
                d_in: int, 
                d_model: int, 
                d_out: int,
                num_encoder_layers: int = 1, 
                max_len: int = 5000, 
                num_heads: int = 1, 
                mask_value: float = 1e-9, 
                activation: str = 'relu', 
                dropout: float = 0.1) -> None:
        super(AttentiveEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        self.module_emb = Embedder(d_in=d_in, d_out=d_model)
        self.module_pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len, batch_first=True)
        self.list_module_transformer_encoder = nn.ModuleList([
            TransformerEncoder(
                d_model=d_model, num_heads=num_heads, mask_value=mask_value, activation=activation, dropout=dropout) 
            for _ in range(num_encoder_layers)])
        self.module_attentive = SelfAttentiveLayer(d_in=d_model, d_model=d_model, d_out=d_model, num_heads=num_heads, dropout=dropout)
        self.f_out = nn.Linear(d_model, d_out)

    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: Tensor, shape [N, T, d_in]
        Returns:
            outputs: Tensor, shape [N, d_out]
            scores: Tensor, shape [N, num_heads, T]
        """
        x = self.module_emb(inputs)
        x = self.module_pe(x * math.sqrt(self.d_model))
        for encoder in self.list_module_transformer_encoder:
            x = encoder(x)
        x, scores = self.module_attentive(x)
        outputs = self.f_out(self.dropout(x))
        return outputs, scores


class AudioGestureSimilarityNet(nn.Module):

    def __init__(self,
                d_audio: int,
                d_motion: int,
                d_model: int, 
                num_encoder_layers: int = 1, 
                max_len: int = 5000, 
                num_heads: int = 1, 
                mask_value: float = 1e-9, 
                activation: str = 'relu', 
                dropout: float = 0.1) -> None:
        super(AudioGestureSimilarityNet, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.module_audio_encoder = AttentiveEncoder(
            d_in=d_audio, 
            d_model=d_model, 
            d_out=d_model, 
            num_encoder_layers=num_encoder_layers, 
            max_len=max_len, 
            num_heads=num_heads, 
            mask_value=mask_value, 
            activation=activation,
            dropout=dropout)

        self.module_motion_encoder = AttentiveEncoder(
            d_in=d_motion,
            d_model=d_model, 
            d_out=d_model,
            num_encoder_layers=num_encoder_layers, 
            max_len=max_len, 
            num_heads=num_heads, 
            mask_value=mask_value, 
            activation=activation,
            dropout=dropout)

    def forward(self, in_audio: Tensor, in_motion: Tensor) -> Tensor:
        """
        Args:
            in_audio: Tensor, shape [batch_size, T, d_audio]
            in_motion: Tensor, shape [batch_size, T, d_motion]
        Returns:
            similarity: Tensor, shape [N, 1]
            scores_audio: Tensor, shape [N, num_heads, T], attention score for audio inputs
            scores_motion: Tensor, shape [N, num_heads, T], attention score for motion inputs
        """
        z_audio, scores_audio = self.module_audio_encoder(in_audio)
        z_motion, scores_motion = self.module_motion_encoder(in_motion)
        similarity = self.calculate_similarity(z_audio, z_motion)
        return similarity, scores_audio, scores_motion

    @staticmethod
    def calculate_similarity(x0: Tensor, x1: Tensor) -> Tensor:
        """
        Args:
            x0: Tensor, shape [batch_size, d_model]
            x1: Tensor, shape [batch_size, d_model]
        Returns:
            outputs: Tensor, shape [N, 1]
        """
        return torch.einsum('bd,bd->b', x0, x1).unsqueeze(1)