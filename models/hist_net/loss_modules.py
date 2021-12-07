from torch import nn
import torch
from torch import Tensor
from torch.nn import functional as F
from jit_gru import JitGRU


def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky':
        return nn.LeakyReLU()


def get_conv_block(d_in, d_out, downsample=False, padding=0, batchnorm=False):
        if downsample:
            kernel_size = 4
            stride = 2
        else:
            kernel_size = 3
            stride = 1
        if batchnorm:
            net = nn.Sequential(
                nn.Conv1d(d_in, d_out, kernel_size, stride, padding),
                nn.BatchNorm1d(d_out))
        else:
            net = nn.Sequential(
                nn.Conv1d(d_in, d_out, kernel_size, stride, padding))
        return net


def continuity_loss(y_pred: Tensor, y_true: Tensor, prev_len: int) -> Tensor:
    # Continuity loss: l1
    loss = F.smooth_l1_loss(y_pred[:, :prev_len], y_true[:, -prev_len:], reduction='none')
    loss = loss.sum(dim=[1, 2]) # sum over joint & time step
    loss = loss.mean() # mean over batch samples
    return loss


class JitBiGRU(nn.Module):

    def __init__(self, d_in: int, d_model: int, num_layers: int = 1, dropout: float = 0.1, batch_first: bool = False, bias: bool = True):
        super(JitBiGRU, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.batch_first = batch_first

        if num_layers == 1:
            self.forward_layers = nn.ModuleList([JitGRU(d_in, d_model)])
            self.backward_layers = nn.ModuleList([JitGRU(d_in, d_model)])
        else:
            self.forward_layers = nn.ModuleList([JitGRU(d_in, d_model)] + [JitGRU(d_model, d_model) for _ in range(num_layers - 1)])
            self.backward_layers = nn.ModuleList([JitGRU(d_in, d_model)] + [JitGRU(d_model, d_model) for _ in range(num_layers - 1)])

    def forward(self, x):
        # TODO return states
        # Handle batch_first cases
        if self.batch_first:
            x = x.permute(1, 0, 2)

        forward_output = x
        backward_output = x

        for i in range(self.num_layers):
            forward_output, _ = self.forward_layers[i](forward_output)
            backward_output, _ = self.backward_layers[i](backward_output)

        output = torch.cat([forward_output, backward_output], dim=-1)

        # Don't forget to handle batch_first cases for the output too!
        if self.batch_first:
            output = output.permute(1, 0, 2)

        return output


class SeqDiscriminator(nn.Module):

    def __init__(self, d_in: int, d_cond: int, hparams: dict,  activation: str = 'relu', dropout: float = 0.1):
        super(SeqDiscriminator, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.f_cond = nn.Linear(d_cond, hparams['dim']//2)
        self.f_input = nn.Linear(d_in, hparams['dim']//2)
        self.f_gru = JitBiGRU(hparams['dim'], hparams['dim'], num_layers=hparams['num_layers'], dropout=dropout, batch_first=True)
        self.f_out = nn.Linear(hparams['dim'], 1)
        self.activation = get_activation(activation)

        self.count_parameters()

    def forward(self, input: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            input: Tensor, shape [batch_size, seq_len, d_input]
            cond: Tensor, shape [batch_size, seq_len, d_cond]
        Returns:
            x: Tensor, scaler
        """

        x_cond = self.activation(self.f_cond(self.dropout(cond)))
        x_input = self.activation(self.f_input(self.dropout(input)))
        x = torch.cat([x_cond, x_input], dim=-1)

        x, _ = self.f_gru(self.dropout(x))
        x = x[:, :, :x.shape[2]//2] + x[:, :, x.shape[2]//2:]  # sum bidirectional outputs
        x = self.f_out(self.dropout(x))
        x = torch.sigmoid(x).squeeze(-1).mean()
        return x

    def count_parameters(self):
        print(f"{self._get_name()} params count: ", sum([p.numel() for p in self.parameters() if p.requires_grad]))


class ChunkDiscriminator(nn.Module):

    def __init__(self, d_in: int, d_cond: int, hparams: dict, activation: str = 'relu', dropout: float = 0.1):
        super(ChunkDiscriminator, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.activation = get_activation(activation)

        self.f_input = nn.Linear(d_in, hparams['dim']//2)
        self.f_cond = nn.Linear(d_cond, hparams['dim']//2)

        self.f_conv = nn.Sequential(
            get_conv_block(hparams['dim'], hparams['dim'], downsample=False, batchnorm=False),
            nn.LayerNorm((128, 32)),
            self.activation,
            self.dropout,
            get_conv_block(hparams['dim'], 2 * hparams['dim'], downsample=True, batchnorm=False),
            get_conv_block(2 * hparams['dim'], 2 * hparams['dim'], downsample=False, batchnorm=False),
            nn.LayerNorm((256, 13)),
            self.activation,
            self.dropout,
            get_conv_block(2 * hparams['dim'], 4 * hparams['dim'], downsample=True, batchnorm=False),
            get_conv_block(4 * hparams['dim'], 4 * hparams['dim'], downsample=False, batchnorm=False),
            nn.LayerNorm((512, 3)),
            self.activation,
            self.dropout,
            nn.Conv1d(4 * hparams['dim'], 8 * hparams['dim'], 3),
            nn.LayerNorm((1024, 1)),) # for 34 frames

        self.f_out = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.LayerNorm(256),
            self.activation,
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            self.activation,
            nn.Linear(64, 1))

        self.count_parameters()

    def count_parameters(self):
        print(f"{self._get_name()} params count: ", sum([p.numel() for p in self.parameters() if p.requires_grad]))

    def forward(self, input: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            input: Tensor, shape [batch_size, chunk_len, d_input]
            cond: Tensor, shape [batch_size, chunk_len, d_cond]
        Returns:
            x: Tensor, scaler
        """
        x_cond = self.activation(self.f_cond(self.dropout(cond)))
        x = self.activation(self.f_input(self.dropout(input)))
        x = torch.cat([x, x_cond], dim=-1)

        x = x.transpose(1, 2)
        x = self.activation(self.f_conv(self.dropout(x)))
        x = x.transpose(1, 2)
        x = x.view(x.shape[0], -1)
        x = self.f_out(self.dropout(x)).mean()
        return x


if __name__ == '__main__':

    MAX_TIME_STEP = 40
    CHUNK_LEN = 34
    PREV_LEN = 4
    D_COND = 2
    D_MOTION = 36
    D_MODEL = 128
    ACTIVATION = 'relu'
    DROPOUT = 0.1
    BATCH_SIZE = 5

    SEQ_DISC_HPARAMS = dict(
        dim = 256,
        num_layers = 2)

    CHUNK_DISC_HPARMAS = dict(
        dim = 128)

    cond_seqs = torch.zeros(BATCH_SIZE, MAX_TIME_STEP, D_COND)
    cond_chunks = torch.zeros(BATCH_SIZE, MAX_TIME_STEP, CHUNK_LEN, D_COND)
    prev_motion_chunks = torch.zeros(BATCH_SIZE, MAX_TIME_STEP, CHUNK_LEN, D_MOTION)

    output_chunks = torch.zeros(BATCH_SIZE, MAX_TIME_STEP, CHUNK_LEN, D_MOTION)
    output_seqs = torch.zeros(BATCH_SIZE, MAX_TIME_STEP, D_MOTION)

    chunk_disc = ChunkDiscriminator(d_in=D_MOTION, d_cond=D_COND, hparams=CHUNK_DISC_HPARMAS, activation=ACTIVATION, dropout=DROPOUT)
    seq_disc = SeqDiscriminator(d_in=D_MOTION, d_cond=D_COND, hparams=SEQ_DISC_HPARAMS, activation=ACTIVATION, dropout=DROPOUT)


    # Compute loss
    output_chunks = output_chunks.view(-1, *output_chunks.shape[2:])
    cond_chunks = cond_chunks.view(-1, *cond_chunks.shape[2:])
    prev_motion_chunks = prev_motion_chunks.view(-1, *prev_motion_chunks.shape[2:])

    loss_chunks = chunk_disc(output_chunks, cond_chunks)
    loss_seq = seq_disc(output_seqs, cond_seqs)
    loss_cl = continuity_loss(output_chunks, prev_motion_chunks, prev_len=PREV_LEN)

    print(loss_chunks.shape, loss_seq.shape, loss_cl.shape)

