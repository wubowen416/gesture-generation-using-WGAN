import numpy as np
from torch.utils.data import Dataset
import torch

def chunkize(x, chunklen, stride=1):
    num_chunk = (x.shape[0] - chunklen) // stride + 1
    return [x[i_chunk * stride:(i_chunk * stride) + chunklen] for i_chunk in range(num_chunk)]

class TrainDataset(Dataset):

    '''X and Y are lists of (T, dim) numpy array'''

    def __init__(self, X, Y, chunklen, seedlen, stride=1):

        # make chunks
        X_chunks = []
        for x in X:
            if x.shape[0] < chunklen:
                continue
            else:
                chunk = chunkize(x, chunklen, stride)
                X_chunks.extend(chunk)
        self.X_chunks = np.array(X_chunks)

        Y_chunks = []
        for x in Y:
            if x.shape[0] < chunklen:
                continue
            else:
                chunk = chunkize(x, chunklen, stride)
                Y_chunks.extend(chunk)
        self.Y_chunks = np.array(Y_chunks)

        self.chunklen = chunklen
        self.seedlen = seedlen

    def __len__(self):
        return len(self.X_chunks)

    def __getitem__(self, idx):

        speech_chunk = self.X_chunks[idx]
        motion_chunk = self.Y_chunks[idx]

        seed_motion = np.zeros((self.chunklen, motion_chunk.shape[-1]))
        seed_motion[:self.seedlen, :] = motion_chunk[:self.seedlen, :]

        seed_motion = torch.from_numpy(seed_motion).float()
        speech_inputs = torch.from_numpy(speech_chunk).float()
        outputs = torch.from_numpy(motion_chunk).float()
        return seed_motion, speech_inputs, outputs
    

class TestDataset(Dataset):

    '''X and Y are lists of (T, dim) numpy array'''

    def __init__(self, X, Y, chunklen, pastlen, stride=1):

        self.X = X
        self.Y = Y

        self.chunklen = chunklen
        self.pastlen = pastlen

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        speech = self.X[idx]
        motion = self.Y[idx]

        speech_chunks = chunkize(speech, self.chunklen, stride=self.chunklen-self.pastlen)
        speech_inputs = torch.from_numpy(speech_chunks).float()
        motion = torch.from_numpy(motion).float()
        return speech_inputs, motion
    