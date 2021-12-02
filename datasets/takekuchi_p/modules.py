import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(self, features, motions, chunk_len, seed_len):

        super().__init__()

        indexs = [i for i, x in enumerate(features) if x.shape[0] > chunk_len + seed_len]
        features = [features[i] for i in indexs]
        motions = [motions[i] for i in indexs]

        self.features = features
        self.motions = motions

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return dict(
            feature = self.features[index],
            motion = self.motions[index]
        )

        