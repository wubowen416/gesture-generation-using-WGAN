import numpy as np
from torch.utils.data import Dataset
import torch


def chunkize(x, chunklen, stride=1):
    num_chunk = (x.shape[0] - chunklen) // stride + 1
    return np.array([x[i_chunk * stride:(i_chunk * stride) + chunklen] for i_chunk in range(num_chunk)])


