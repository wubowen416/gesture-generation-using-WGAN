import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from jit_gru import JitGRU


def get_linear_block(input_size, output_size):
    return nn.Sequential(
            nn.Linear(input_size, output_size//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(output_size//2, output_size))


class Mixer(nn.Module):

    def __init__(self, cond_size, output_size, param):
        super().__init__()

        self.cond_size = cond_size
        self.output_size = output_size
        self.hidden_size = param['hidden_size']
        self.noise_size = param['noise_size']
        self.num_layers = param['num_layers']
        self.device = param['device']
        self.dropout = param['dropout']
        self.bidirectional = param['bidirectional']

        # Modules
        self.mp = get_linear_block(self.cond_size, self.hidden_size)
        self.ne = get_linear_block(self.noise_size, self.hidden_size)
        self.mix = nn.GRU(2*self.hidden_size, 2*self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional, dropout=self.dropout, batch_first=True)
        self.proj = get_linear_block(2*self.hidden_size, output_size)
        self.drop = nn.Dropout(p=self.dropout, inplace=True)
    
    def forward(self, cond):
        B, N, T, _ = cond.size()

        x_mp = self.mp(cond.view(B*N, T, -1))
        x_mp = F.layer_norm(x_mp, x_mp.size()[1:])
        x_mp = self.drop(x_mp)

        x_ne = self.ne(self.get_noise(B*N, T))
        x_ne = F.layer_norm(x_ne, x_ne.size()[1:])
        x_ne = self.drop(x_ne)

        x = torch.cat([x_mp, x_ne], dim=2)

        x, _ = self.mix(x)
        x = F.layer_norm(x, x.size()[1:])
        x = x[:, :, :x.size(2)//2] + x[:, :, x.size(2)//2:]  # sum bidirectional outputs
        x = self.drop(x)

        x = self.proj(x)
        x = x.view(B, N, T, -1)
        return x

    
    def get_noise(self, batch_size, time_step):
        return torch.normal(mean=0, std=1, size=(batch_size, 1, self.noise_size), device=self.device).repeat(1, time_step, 1)

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class HistNet(nn.Module):
    
    def __init__(self, cond_size, output_size, hidden_size, chunk_size, prev_size, smoothing, mixer_param, num_layers=1, dropout=0, device='cpu'):

        super().__init__()

        self.cond_size = cond_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.prev_size = prev_size
        self.smoothing = smoothing
        self.num_layers = num_layers
        self.device = device

        self.mixer = Mixer(cond_size=hidden_size, output_size=output_size, param=mixer_param)

        self.ce = get_linear_block(cond_size, hidden_size//2)
        self.me = get_linear_block(output_size, hidden_size//2)
        self.memo = nn.GRU(chunk_size*hidden_size, chunk_size*hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)


    def forward(self, cond, m_prev, num_chunks, lengths, hs=None):

        B, N, T, _ = cond.size()

        x_ce = self.ce(cond)
        x_me = self.padding(self.me(m_prev), T)

        x = torch.cat([x_ce, x_me], dim=3)
        x = x.view(B, N, -1)

        if hs is None:
            hs = torch.zeros(1*self.num_layers, B, x.size(2))

        x_packed = pack_padded_sequence(x, num_chunks, batch_first=True)

        x, hs = self.memo(x_packed, hs)
        x, _ = pad_packed_sequence(x_packed, batch_first=True)

        x = x.contiguous().view(B, N, T, -1)

        x = self.mixer(cond=x)

        if self.smoothing == 'interpolation':
            x = self.interpolate(x)

        # Set padding parts to zero
        for i, l in enumerate(lengths):
            x[i, int(l):] = 0
        return x

    def interpolate(self, x):
        ''' Interpolate between chunks.
            x (B, N_chunk, T, *)
        '''
        output = x[:, 0][:self.chunk_size-self.prev_size]
        n_chunk = x.size(1)
        for i in range(1, n_chunk):
            trans_prev = x[:, i-1][:, -self.prev_size:]
            trans_next = x[:, i][:, :self.prev_size]
            # Transition strategy
            trans_motion = []
            for k in range(self.prev_size):
                trans = ((self.prev_size - k) / (self.prev_size + 1)) * trans_prev[:, k] + ((k + 1) / (self.prev_size + 1)) * trans_next[:, k]
                trans_motion.append(trans.unsqueeze(1))
            trans_motion = torch.cat(trans_motion, dim=1)
            # Append each
            if i != n_chunk - 1: # not last chunk
                output = torch.cat([output, trans_motion, x[:, i][:, self.prev_size:self.chunk_size-self.prev_size]], dim=1)
            else: # last chunk
                output = torch.cat([output, trans_motion, x[:, i][:, self.prev_size:self.chunk_size]], dim=1)
        return output

    def padding(self, x, time_step):
        padding = torch.zeros(x.size(0), x.size(1), time_step-x.size(2), x.size(3)).to(self.device)
        return torch.cat([x, padding], dim=2)


class Discriminator(nn.Module):

    def __init__(self, cond_size, output_size, hidden_size, num_layers=1, bidirectional=True, dropout=0):
        super().__init__()

        self.ce = get_linear_block(cond_size, hidden_size//2)
        self.me = get_linear_block(output_size, hidden_size//2)
        self.gru = JitGRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, cond, input):

        x_cond = self.ce(cond)
        x_input = self.me(input)

        x = torch.cat([x_cond, x_input], dim=-1)

        x, _ = self.gru(x)

        x = x.mean(dim=[1, 2])

        return x
        




if __name__ == '__main__':

    cond = torch.zeros(5, 20, 34, 2)
    m_prev = torch.zeros(5, 20, 4, 36)
    num_chunks = torch.Tensor([20, 20, 10, 5, 3])
    lengths = 2 * 4 + (34 - 4) * num_chunks

    mixer_param = dict(
        hidden_size = 256,
        noise_size = 20,
        num_layers = 2,
        bidirectional = True,
        dropout = 0,
        device = 'cpu'
    )

    net = HistNet(cond_size=2, output_size=36, hidden_size=128, chunk_size=34, prev_size=4, smoothing='interpolation', mixer_param=mixer_param)
    disc = Discriminator(cond_size=2, output_size=36, hidden_size=256)

    outputs = net(cond, m_prev, num_chunks, lengths)

    cond_ori = torch.zeros(5, outputs.size(1), 2)

    disc(cond_ori, outputs)