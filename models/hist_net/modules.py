import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math


def get_linear_block(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, output_size//2),
        nn.LeakyReLU(0.2, True),
        nn.Linear(output_size//2, output_size))

def get_conv_block(in_channels, out_channels, downsample=False, padding=0, batchnorm=False):
        if downsample:
            kernel_size = 4
            stride = 2
        else:
            kernel_size = 3
            stride = 1
            
        if batchnorm:
            net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.2, True)
            )

        else:
            net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
                nn.LeakyReLU(0.2, True)
            )
        return net


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

        if len(cond.size()) == 4:
            B, N, T, _ = cond.size()
            B_new = B * N
        else:
            B, T, _ = cond.size()
            B_new = B

        x_mp = self.mp(cond.view(B_new, T, -1))
        x_mp = F.layer_norm(x_mp, x_mp.size()[1:])
        x_mp = self.drop(x_mp)

        x_ne = self.ne(self.get_noise(B_new, T))
        x_ne = F.layer_norm(x_ne, x_ne.size()[1:])
        x_ne = self.drop(x_ne)

        x = torch.cat([x_mp, x_ne], dim=2)

        x, _ = self.mix(x)
        x = F.layer_norm(x, x.size()[1:])
        x = x[:, :, :x.size(2)//2] + x[:, :, x.size(2)//2:]  # sum bidirectional outputs
        x = self.drop(x)

        x = self.proj(x)

        if len(cond.size()) == 4:
            x = x.view(B, N, T, -1)
        return x

    
    def get_noise(self, batch_size, time_step):
        return torch.normal(mean=0, std=1, size=(batch_size, 1, self.noise_size), device=self.device).repeat(1, time_step, 1)

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class RNNHistNet(nn.Module):
    
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


    def forward(self, cond_speech, cond_motion, num_chunks, hs=None):
        '''
        Returns:
            x - Model output, shape (B, N, T, d)
            x_seq - Interpolated x, shape (B, T, d)
            hs - Hidden state of memo
        '''

        B, N, T, _ = cond_speech.size()

        x_ce = self.ce(cond_speech)
        x_me = self.me(cond_motion)

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
            x_itp = self.interpolate(x)

        return x, x_itp, hs

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

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class QueryKeyValue(nn.Module):

    def __init__(self, input_0_size, input_1_size, output_size, num_layers=1, bidirectional=True, dropout=0):
        super().__init__()

        self.gru_0 = nn.GRU(input_0_size, output_size//2, bidirectional=bidirectional, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.gru_1 = nn.GRU(input_1_size, output_size//2, bidirectional=bidirectional, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.wq = nn.Parameter(data=torch.Tensor(output_size, output_size), requires_grad=True)
        self.wk = nn.Parameter(data=torch.Tensor(output_size, output_size), requires_grad=True)
        self.wv = nn.Parameter(data=torch.Tensor(output_size, output_size), requires_grad=True)
        self._init_weights()

    def _init_weights(self):
        stdv = 1. / math.sqrt(self.wq.size(1))
        self.wq.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.wk.size(1))
        self.wk.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.wv.size(1))
        self.wv.data.uniform_(-stdv, stdv)

    def forward(self, input_0, input_1):
        x_0, _ = self.gru_0(input_0)
        x_0 = x_0[:, :, :x_0.size(2)//2] + x_0[:, :, x_0.size(2)//2:] # sum bi-directional
        x_0 = x_0.mean(dim=1)

        x_1, _ = self.gru_1(input_1)
        x_1 = x_1[:, :, :x_1.size(2)//2] + x_1[:, :, x_1.size(2)//2:] # sum bi-directional
        x_1 = x_1.mean(dim=1)

        x = torch.cat([x_0, x_1], dim=-1)
        
        query = x @ self.wq
        key = x @ self.wk
        value = x @ self.wv
        return query, key, value


class AttentionHistNet(nn.Module):

    def __init__(self, input_0_size, input_1_size, hidden_size, output_size, mixer_param, container_size=10, num_layers=1, bidirectional=True, dropout=0, device='cpu'):
        super().__init__()

        self.container_size = container_size
        self.hidden_size = hidden_size
        self.device = device

        self.f_input_0 = nn.Linear(input_0_size, hidden_size//2)
        self.f_input_1 = nn.GRU(input_1_size, hidden_size//2, bidirectional=bidirectional, num_layers=num_layers, dropout=dropout, batch_first=True)

        self.f_qkv = QueryKeyValue(input_0_size, input_1_size, hidden_size)
        self.f_mix = Mixer(hidden_size, output_size, mixer_param)


    def forward(self, input_0_chunk, input_1_chunk):

        B, N, T, _ = input_0_chunk.size()

        self.container_key = torch.zeros(B, self.container_size, self.hidden_size).to(self.device)
        self.container_value = torch.zeros(B, self.container_size, self.hidden_size).to(self.device)
        self.memo_strength = torch.ones(B, self.container_size).to(self.device)

        for step in range(N):

            input_0 = input_0_chunk[:, step]
            input_1 = input_1_chunk[:, step]

            output = self.step(input_0, input_1)

            print(output.shape)


    def step(self, input_0, input_1):

        # Current step information
        x_0 = self.f_input_0(input_0)

        x_1, _ = self.f_input_1(input_1)
        x_1 = x_1[:, :, :x_1.size(2)//2] + x_1[:, :, x_1.size(2)//2:] # sum bi-directional
        x_1 = x_1.mean(dim=1) # mean over time dim

        x = torch.cat([x_0, x_1.unsqueeze(1).repeat(1, x_0.size(1), 1)], dim=-1)

        print("x size: ", x.size())

        # Attention information
        query, key, value = self.f_qkv(input_0, input_1)

        # Compute weights
        weights = torch.bmm(query.unsqueeze(1), self.container_key.transpose(1, 2))
        weights = F.softmax(weights / torch.sqrt(torch.Tensor([self.hidden_size]).to(self.device)), dim=2).squeeze(1)

        # Update memo
        self.update_memo_strength(value)

        # Weighted sum
        memory = (.5 * (weights + self.memo_strength).unsqueeze(2) * self.container_value).sum(dim=1)

        # Residual
        x += memory.unsqueeze(1)

        # Pose estimator
        out = self.f_mix(x)

        # Update containers



        return out

    def update_container(self, key, value):

        

    def update_memo_strength(self, value):
        cos_sim = F.cosine_similarity(value.unsqueeze(1), self.container_value, dim=2)
        amplitude = F.sigmoid(cos_sim) + torch.Tensor([0.5]).to(self.device)
        self.memo_strength *= amplitude




class SeqDiscriminator(nn.Module):

    def __init__(self, input_size, cond_size, hidden_size, num_layers=1, bidirectional=True, dropout=0):
        super().__init__()

        self.ce = get_linear_block(cond_size, hidden_size//2)
        self.me = get_linear_block(input_size, hidden_size//2)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input, cond):

        x_cond = self.ce(cond)
        x_input = self.me(input)

        x = torch.cat([x_cond, x_input], dim=-1)

        x, _ = self.gru(x)
        x = x[:, :, :x.size(2)//2] + x[:, :, x.size(2)//2:]  # sum bidirectional outputs
        x = self.out(x)
        return torch.sigmoid(x).squeeze(-1)

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class ChunkDiscriminator(nn.Module):

    def __init__(self, input_size, cond_size, hidden_size):
        super().__init__()

        self.cond_size = cond_size
        self.input_size = input_size

        self.ae = get_linear_block(cond_size, hidden_size)
        self.me = get_linear_block(input_size, hidden_size)
        self.pe = get_linear_block(input_size, hidden_size)

        self.conv_layers = nn.Sequential(
            get_conv_block(3*hidden_size, 3*hidden_size, downsample=False, batchnorm=False),
            nn.LayerNorm((192, 32)),
            get_conv_block(3*hidden_size, 3*2 * hidden_size, downsample=True, batchnorm=False),
            nn.LayerNorm((384, 15)),
            get_conv_block(3*2 * hidden_size, 3*2 * hidden_size, downsample=False, batchnorm=False),
            nn.LayerNorm((384, 13)),
            get_conv_block(3*2 * hidden_size, 3*4 * hidden_size, downsample=True, batchnorm=False),
            nn.LayerNorm((768, 5)),
            get_conv_block(3*4 * hidden_size, 3*4 * hidden_size, downsample=False, batchnorm=False),
            nn.LayerNorm((768, 3)),
            nn.Conv1d(3*4 * hidden_size, 3*8 * hidden_size, 3)
        ) # for 34 frames

        self.out_net = nn.Sequential(
            nn.Linear(1536, 256), 
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
        )

    def forward(self, input, cond_speech, cond_motion):
        x_ae = self.ae(cond_speech)
        x_pe = self.pe(cond_motion)
        x = self.me(input)
        x = torch.cat([x_ae, x_pe, x], dim=-1) # (N, T, hidden_size)

        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), -1)

        return self.out_net(x).mean()

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


def test_rnn_hist():

    CHUNK_SIZE = 34
    COND_SPEECH_SIZE = 2 # speech dimension + motion dimension
    COND_MOTION_SIZE = 36
    MOTION_SIZE = 36
    BATCH_SIZE = 5
    PREV_SIZE = 4
    BTACH_SIZE = 5

    # ---------- Inputs
    conds_speech_seq = torch.zeros(BTACH_SIZE, 608, COND_SPEECH_SIZE)
    labels_seq = torch.ones(BTACH_SIZE, 608)
    conds_speech_chunk = torch.zeros(BTACH_SIZE, 20, CHUNK_SIZE, COND_SPEECH_SIZE)
    conds_motion_chunk = torch.zeros(BTACH_SIZE, 20, CHUNK_SIZE, COND_MOTION_SIZE)
    lengths_chunk = torch.Tensor([20, 20, 10, 5, 3])

    lengths_seq = 2 * PREV_SIZE + (CHUNK_SIZE - PREV_SIZE) * lengths_chunk
    masks_seq = torch.zeros(*conds_speech_seq.size()[:-1])
    for i, l in enumerate(lengths_seq):
        masks_seq[i, :int(l)] = 1 / int(l)
    # ---------

    mixer_param = dict(
        hidden_size = 256,
        noise_size = 20,
        num_layers = 2,
        bidirectional = True,
        dropout = 0,
        device = 'cpu'
    )

    rnn_hist_net = RNNHistNet(cond_size=2, output_size=36, hidden_size=64, chunk_size=34, prev_size=4, smoothing='interpolation', mixer_param=mixer_param)
    seq_disc = SeqDiscriminator(input_size=36, cond_size=2, num_layers=2, hidden_size=512)
    chunk_disc = ChunkDiscriminator(input_size=36, cond_size=2, hidden_size=64)

    outputs_chunk, outputs_itped, hss = rnn_hist_net(conds_speech_chunk, conds_motion_chunk, lengths_chunk, hs=None)

    print("HistNet params count: ", rnn_hist_net.count_parameters())
    print("ChunkDisc params count: ", chunk_disc.count_parameters())
    print("SeqDisc params count: ", seq_disc.count_parameters())

    def remove_padding(x, lengths):
        return torch.cat([x[i, :int(l)] for i, l in enumerate(lengths)], dim=0)

    # Chunk loss: adv
    # Remove padded chunks
    outputs_chunk = remove_padding(outputs_chunk, lengths_chunk)
    cond_speech = remove_padding(conds_speech_chunk, lengths_chunk)
    cond_motion = remove_padding(conds_motion_chunk, lengths_chunk)
    loss_chunk = chunk_disc(outputs_chunk, cond_speech, cond_motion)
    print("chunk loss: adv ", loss_chunk.item())

    # Continuity loss: l1
    loss_cl = F.smooth_l1_loss(outputs_chunk[:, :PREV_SIZE], cond_motion[:, -PREV_SIZE:], reduction='none')
    loss_cl = loss_cl.sum(dim=[1, 2]) # sum over joint & time step
    loss_cl = loss_cl.mean() # mean over batch samples
    print(loss_cl.item())

    # Seq loss: adv
    y_pred = seq_disc(outputs_itped, conds_speech_seq)
    loss_seq = F.binary_cross_entropy(y_pred, labels_seq, reduction='none')
    # Only compute loss for no padded parts by multiplying mask
    loss_seq = torch.trace(loss_seq @ masks_seq.T) / masks_seq.size(0)
    print(loss_seq.item())

    

if __name__ == '__main__':

    # test_rnn_hist()

    ###
    CHUNK_SIZE = 34
    COND_SPEECH_SIZE = 2 # speech dimension + motion dimension
    COND_MOTION_SIZE = 36
    MOTION_SIZE = 36
    BATCH_SIZE = 5
    PREV_SIZE = 4
    BTACH_SIZE = 5

    mixer_param = dict(
        hidden_size = 256,
        noise_size = 20,
        num_layers = 2,
        bidirectional = True,
        dropout = 0,
        device = 'cpu'
    )

    # ---------- Inputs
    conds_speech_seq = torch.zeros(BTACH_SIZE, 608, COND_SPEECH_SIZE)
    labels_seq = torch.ones(BTACH_SIZE, 608)
    conds_speech_chunk = torch.zeros(BTACH_SIZE, 20, CHUNK_SIZE, COND_SPEECH_SIZE)
    conds_motion_chunk = torch.zeros(BTACH_SIZE, 20, CHUNK_SIZE, COND_MOTION_SIZE)
    lengths_chunk = torch.Tensor([20, 20, 10, 5, 3])

    lengths_seq = 2 * PREV_SIZE + (CHUNK_SIZE - PREV_SIZE) * lengths_chunk
    masks_seq = torch.zeros(*conds_speech_seq.size()[:-1])
    for i, l in enumerate(lengths_seq):
        masks_seq[i, :int(l)] = 1 / int(l)
    # --------------------------------------


    net = AttentionHistNet(input_0_size=COND_SPEECH_SIZE, input_1_size=COND_MOTION_SIZE, hidden_size=128, output_size=MOTION_SIZE, mixer_param=mixer_param)

    net(conds_speech_chunk, conds_motion_chunk)