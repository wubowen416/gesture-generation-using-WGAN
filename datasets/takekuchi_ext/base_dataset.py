from cProfile import label
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
import torch

def chunkize(x, chunklen, stride=1):
    num_chunk = (x.shape[0] - chunklen) // stride + 1
    return np.array([x[i_chunk * stride:(i_chunk * stride) + chunklen] for i_chunk in range(num_chunk)])


def kmeans_clustering(position_data, category, k):

    # Extract features from data
    def velocity_sum(sample):
        '''sample of shape (T, dim)
        Total velocity of all joints
        '''
        return np.sum(np.abs(sample[1:] - sample[:-1])) / sample.shape[0]

    def avg_hand_amplitude(sample):
        '''sample of shape (T, dim)
        Amplitude of hand position. [8] is left hand, [11] is right hand.
        '''
        sample = sample.reshape(-1, 12, 3)
        # Calculate distance pairwisely
        left_hand_data = sample[:][:, 8, :]
        right_hand_data = sample[:][:, 11, :]

        def pairwise_distance(data):
            data_a = np.repeat(data[:, np.newaxis, :], data.shape[0], axis=1)
            data_b = np.transpose(data_a, [1, 0, 2])
            return np.sqrt(np.sum((data_a - data_b)**2, axis=-1))

        left_amp_max = np.max(pairwise_distance(left_hand_data))
        right_amp_max = np.max(pairwise_distance(right_hand_data))
        avg_max_amp = 0.5 * (left_amp_max + right_amp_max)
        return avg_max_amp

    def extract_features(sample):
        vel = velocity_sum(sample)
        amp = avg_hand_amplitude(sample)
        return [vel, amp]

    features = np.array(list(map(extract_features, position_data)))
    features = StandardScaler().fit_transform(features)

    # Cluster use K-means
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=20).fit(features)

    # Plot clustering result
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('Agg')
    # import seaborn as sns
    # sns.set()
    # colors = [['r','g','b'][label] for label in kmeans.labels_]
    # fig = plt.figure(dpi=100)
    # plt.scatter(features[:, 0], features[:, 1], c=colors, alpha=0.1)
    # plt.tight_layout()
    # plt.xlabel("Amplitude")
    # plt.ylabel("Velocity")
    # plt.title("K-means clustered result")
    # plt.savefig("kmeans_clustered_result.jpg")

    # Rename cluster index to category names
    centers = kmeans.cluster_centers_
    centers_norm = np.sum(centers, axis=1)
    labels = [x for _, x in sorted(zip(centers_norm, list(range(k))))]

    label_dict = {}
    label_dict["low"] = labels[0]
    label_dict["mid"] = labels[1]
    label_dict["high"] = labels[2]

    # Switch category name and index
    # new_dict: 0: "high", 1:"middle", 2:"low"
    new_dict = {}
    for key, value in label_dict.items():
        new_dict[value] = key
    category_labels = [[new_dict[index]] for index in kmeans.labels_]

    category_list = list(category.to_dict().keys())
    enc = OneHotEncoder(categories=[category_list], sparse=False)
    onehot_lables = enc.fit_transform(category_labels)
    return onehot_lables, enc


class TrainDataset(Dataset):

    '''X and Y are lists of (T, dim) numpy array'''

    def __init__(self, X, Y, position_data, chunklen, seedlen, category, stride=1, k=3):

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
        for y in Y:
            if y.shape[0] < chunklen:
                continue
            else:
                chunk = chunkize(y, chunklen, stride)
                Y_chunks.extend(chunk)
        self.Y_chunks = np.array(Y_chunks)

        pos_chunks = []
        for pos in position_data:
            if pos.shape[0] < chunklen:
                continue
            else:
                chunk = chunkize(pos, chunklen, stride)
                pos_chunks.extend(chunk)
        pos_chunks = np.array(pos_chunks)

        # K-means clustering
        category_labels, self.category_encoder = kmeans_clustering(pos_chunks, category, k=k) # (N, n_classes)
        category_vectors = np.repeat(category_labels[:, np.newaxis, :], chunklen, axis=1)

        # Concate to speech features
        self.X_chunks = np.concatenate([self.X_chunks, category_vectors], axis=-1)
        # print(self.X_chunks.shape)
        # print(self.X_chunks[0])

        self.chunklen = chunklen
        self.seedlen = seedlen
        self.k = k

    def get_category_encoder(self):
        return self.category_encoder

    def __len__(self):
        return len(self.X_chunks)

    def __getitem__(self, idx):

        speech_chunk = self.X_chunks[idx]
        motion_chunk = self.Y_chunks[idx]

        seed_motion = np.zeros_like(motion_chunk)
        seed_motion[:self.seedlen, :] = motion_chunk[:self.seedlen, :]

        seed_motion = torch.from_numpy(seed_motion).float()
        speech_inputs = torch.from_numpy(speech_chunk).float()
        outputs = torch.from_numpy(motion_chunk).float()
        return seed_motion, speech_inputs, outputs
    

class TestDataset(Dataset):

    '''X and Y are lists of (T, dim) numpy array'''

    def __init__(self, X, Y, chunklen, pastlen, stride=1, k=3):

        # self.X = X
        # self.Y = Y
        # One sample for each category
        onehot_label = np.eye(k)
        inputs = []
        for x in X:
            chunks = chunkize(x, chunklen, stride=chunklen-pastlen)
            # For every category
            for i in range(k):
                # Add onehot label to each chunk
                new_chunks = np.empty(shape=(0, chunklen, chunks.shape[-1]+k))
                for j in range(chunks.shape[0]):
                    label = onehot_label[i]
                    new_chunk = np.concatenate([chunks[j], np.repeat(label[np.newaxis, :], chunklen, axis=0)], axis=-1)
                    new_chunks = np.concatenate([new_chunks, new_chunk[np.newaxis, :, :]], axis=0)
                inputs.append(new_chunks)

        self.X = inputs
        self.Y = np.repeat(np.array(Y, dtype=object), k)

        self.k = k
        self.chunklen = chunklen
        self.pastlen = pastlen

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        inputs = self.X[idx]
        outputs = self.Y[idx]
        
        inputs = torch.from_numpy(inputs).float()
        outputs = torch.from_numpy(outputs).float()
        return inputs, outputs
    

class SeqDataset(Dataset):

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