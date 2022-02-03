import numpy as np
import os
import pickle
import joblib as jl
from sklearn.model_selection import train_test_split
from .base_dataset import SeqDataset, TrainDataset, TestDataset
from .lowpass_filter import lowpass_filter
from .utils import transform, line_prepender

class TakekuchiExtDataset:

    def __init__(self, hparams, is_training):

        data_dir = hparams.Dir.processed
        
        # Load scalers
        self.speech_scaler = jl.load(os.path.join(data_dir, 'speech_scaler.jl'))
        self.motion_scaler = jl.load(os.path.join(data_dir, 'motion_scaler.jl'))
        

        # N of categories
        k = len(hparams.Data.extraversion.to_dict().keys())

        if is_training:

            # Load data and scale
            with open(os.path.join(data_dir, "X_train.p"), 'rb') as f:
                train_input = pickle.load(f)
            with open(os.path.join(data_dir, "Y_train.p"), 'rb') as f:
                train_output = pickle.load(f)
            with open(os.path.join(data_dir, "position_train.p"), "rb") as f:
                train_position = pickle.load(f)

            train_input = list(map(self.speech_scaler.transform, train_input))
            train_output = list(map(self.motion_scaler.transform, train_output))

            # Split validation 
            # train_input, val_input, train_output, val_output = train_test_split(
            #     train_input, train_output, test_size=hparams.Data.valid_ratio, random_state=2021)

            # Create pytorch dataset
            self.train_dataset = TrainDataset(train_input, train_output, train_position, hparams.Data.chunklen, hparams.Data.seedlen, hparams.Data.extraversion, stride=1, k=k)
            # self.val_dataset = TrainDataset(val_input, val_output,  hparams.Data.chunklen, hparams.Data.seedlen, stride=1)

            category_encoder = self.train_dataset.get_category_encoder()

            jl.dump(category_encoder, os.path.join(data_dir, 'category_encoder.jl'))

            # Load dev data
            with open(os.path.join(data_dir, "X_dev.p"), 'rb') as f:
                dev_input = pickle.load(f)
            with open(os.path.join(data_dir, "Y_dev.p"), 'rb') as f:
                dev_output = pickle.load(f)

            # scale
            dev_input = list(map(self.speech_scaler.transform, dev_input))
            dev_output = list(map(self.motion_scaler.transform, dev_output))
            self.dev_dataset = TestDataset(dev_input, dev_output, hparams.Data.chunklen, hparams.Data.seedlen, k=k)

        else:
            self.category_encoder = jl.load(os.path.join(data_dir, 'category_encoder.jl'))
            # Load test data
            with open(os.path.join(data_dir, "X_dev.p"), 'rb') as f:
                test_input = pickle.load(f)
            with open(os.path.join(data_dir, "Y_dev.p"), 'rb') as f:
                test_output = pickle.load(f)

            test_input = list(map(self.speech_scaler.transform, test_input))
            test_output = list(map(self.motion_scaler.transform, test_output))
            self.test_dataset = TestDataset(test_input, test_output, hparams.Data.chunklen, hparams.Data.seedlen, k=k)

        self.input_dim = self.speech_scaler.mean_.shape[0] + k
        self.output_dim = self.motion_scaler.mean_.shape[0]
            

    def get_train_dataset(self):
        return self.train_dataset

    def get_dev_dataset(self):
        return self.dev_dataset
    
    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset
        
    def get_dims(self):
        return self.input_dim, self.output_dim

    def get_scaler(self):
        return self.speech_scaler, self.motion_scaler

    def encode_category(self, cate):
        return self.category_encoder.transform(cate)

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