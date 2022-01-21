import numpy as np
import os
import pickle
import joblib as jl
from sklearn.model_selection import train_test_split
from .base_dataset import TrainDataset, TestDataset
from .lowpass_filter import lowpass_filter
from .utils import transform, line_prepender

class TakekuchiDataset:

    def __init__(self, hparams, is_training):

        data_dir = hparams.Dir.processed

        self.chunklen = hparams.Data.chunklen
        self.seedlen = hparams.Data.seedlen
        self.num_sample_per_generator = hparams.Data.num_sample_per_generator
        
        # Load scalers
        self.speech_scaler = jl.load(os.path.join(data_dir, 'speech_scaler.jl'))
        self.motion_scaler = jl.load(os.path.join(data_dir, 'motion_scaler.jl'))

        if is_training:

            # Load data and scale
            with open(os.path.join(data_dir, "X_train.p"), 'rb') as f:
                train_input = pickle.load(f)
            with open(os.path.join(data_dir, "Y_train.p"), 'rb') as f:
                train_output = pickle.load(f)

            self.train_input = list(map(self.speech_scaler.transform, train_input))
            self.train_output = list(map(self.motion_scaler.transform, train_output))

            # Split validation 
            # train_input, val_input, train_output, val_output = train_test_split(
            #     train_input, train_output, test_size=hparams.Data.valid_ratio)

            # Create pytorch dataset
            # self.train_dataset = TrainDataset(train_input[:2], train_output[:2], hparams.Data.chunklen, hparams.Data.seedlen, stride=1)
            # self.val_dataset = TrainDataset(val_input, val_output,  hparams.Data.chunklen, hparams.Data.seedlen, stride=1)

            # Load dev data
            with open(os.path.join(data_dir, "X_dev.p"), 'rb') as f:
                dev_input = pickle.load(f)
            with open(os.path.join(data_dir, "Y_dev.p"), 'rb') as f:
                dev_output = pickle.load(f)

            # scale
            dev_input = list(map(self.speech_scaler.transform, dev_input))
            dev_output = list(map(self.motion_scaler.transform, dev_output))
            self.dev_dataset = TestDataset(dev_input, dev_output, hparams.Data.chunklen, hparams.Data.seedlen)

        else:
            # Load test data
            with open(os.path.join(data_dir, "X_dev.p"), 'rb') as f:
                test_input = pickle.load(f)
            with open(os.path.join(data_dir, "Y_dev.p"), 'rb') as f:
                test_output = pickle.load(f)

            test_input = list(map(self.speech_scaler.transform, test_input))
            test_output = list(map(lowpass_filter, test_output))
            test_output = list(map(self.motion_scaler.transform, test_output))
            # Save test output motion to unity format
            if not os.path.exists('./synthesized/dev_motion'):
                os.makedirs('./synthesized/dev_motion')
                for i, motion in enumerate(test_output):
                    self.save_unity_result(motion, f"./dev_motion/motion_{i}.txt")
            self.test_dataset = TestDataset(test_input, test_output, hparams.Data.chunklen, hparams.Data.seedlen)

        self.speech_dim = self.speech_scaler.mean_.shape[0]
        self.motion_dim = self.motion_scaler.mean_.shape[0]
    
    def train_dataset_generator(self, chunklen, seedlen, stride, max_num_samples):
        num_samples = len(self.train_input)
        if num_samples <= max_num_samples:
            num_dataset = 1
        else:
            if num_samples % max_num_samples == 0:
                num_dataset = num_samples // max_num_samples
            else:
                num_dataset = num_samples // max_num_samples + 1
        for i in range(num_dataset):
            inputs = self.train_input[i*max_num_samples:(i+1)*max_num_samples]
            outputs = self.train_output[i*max_num_samples:(i+1)*max_num_samples]
            yield TrainDataset(inputs, outputs, chunklen, seedlen, stride=stride)

    def get_train_dataset(self):
        return self.train_dataset_generator(self.chunklen, self.seedlen, stride=1, max_num_samples=self.num_sample_per_generator)
        # return self.train_dataset

    def get_dev_dataset(self):
        return self.dev_dataset
    
    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset
        
    def get_dims(self):
        return self.speech_dim, self.motion_dim

    def get_scaler(self):
        return self.speech_scaler, self.motion_scaler

    def save_unity_result(self, data, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # rescale
        data = self.motion_scaler.inverse_transform(data)
        # Unitize
        unity_lines = transform(data)
        unity_lines = lowpass_filter(unity_lines)
        # Save
        np.savetxt(save_path, unity_lines.T)
        prepend_line = f"{data.shape[0]}\n12\n0.05"
        line_prepender(save_path, prepend_line)