import numpy as np
import os
import pickle
import joblib as jl
from .base_dataset import TrainDataset, TestDataset
from .unityfier import Unityfier

import sys
module_path = os.path.abspath(os.path.join('./datasets/genea/data_processing'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pymo.writers import *

class GeneaDataset:

    def __init__(self, hparams, is_training):

        data_dir = hparams.Dir.processed
        self.unityfier = Unityfier()
        
        # Load scalers
        self.speech_scaler = jl.load(os.path.join(data_dir, 'speech_scaler.jl'))
        self.motion_scaler = jl.load(os.path.join(data_dir, 'motion_scaler.jl'))
        self.motiondata_pipe = jl.load(os.path.join(data_dir, 'motiondata_pipe.jl'))

        if is_training:
            # Load data and scale
            with open(os.path.join(data_dir, "X_train.p"), 'rb') as f:
                train_input = pickle.load(f)
            with open(os.path.join(data_dir, "Y_train.p"), 'rb') as f:
                train_output = pickle.load(f)

            train_input = list(map(self.speech_scaler.transform, train_input))
            train_output = list(map(self.motion_scaler.transform, train_output))

            # Create pytorch dataset
            self.train_dataset = TrainDataset(train_input, train_output, hparams.Data.chunklen, hparams.Data.seedlen, stride=1)

            # Load dev data
            with open(os.path.join(data_dir, "X_test.p"), 'rb') as f:
                dev_input = pickle.load(f)
            with open(os.path.join(data_dir, "Y_test.p"), 'rb') as f:
                dev_output = pickle.load(f)

            # scale
            dev_input = list(map(self.speech_scaler.transform, dev_input))
            dev_output = list(map(self.motion_scaler.transform, dev_output))
            self.dev_dataset = TestDataset(dev_input, dev_output, hparams.Data.chunklen, hparams.Data.seedlen)

        else:
            # Load test data
            with open(os.path.join(data_dir, "X_test.p"), 'rb') as f:
                test_input = pickle.load(f)
            with open(os.path.join(data_dir, "Y_test.p"), 'rb') as f:
                test_output = pickle.load(f)

            test_input = list(map(self.speech_scaler.transform, test_input))
            test_output = list(map(self.motion_scaler.transform, test_output))
            self.test_dataset = TestDataset(test_input, test_output, hparams.Data.chunklen, hparams.Data.seedlen)

        self.speech_dim = self.speech_scaler.mean_.shape[0]
        self.motion_dim = self.motion_scaler.mean_.shape[0]
            

    def get_train_dataset(self):
        return self.train_dataset

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

    def save_result(self, data, save_path):
        self.save_unity_result(data, save_path + '.txt')
        self.save_bvh_result(data, save_path + '.bvh')

    def save_unity_result(self, data, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
        # rescale
        data = self.motion_scaler.inverse_transform(data)
        # Unitize
        unity_lines = self.unityfier.transform(data)
        unity_lines = self.unityfier.filter(unity_lines)
        # Save
        np.savetxt(save_path, unity_lines.T)
        prepend_line = f"{data.shape[0]}\n12\n0.05"
        self.unityfier.line_prepender(save_path, prepend_line)

    def save_bvh_result(self, data, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
        # rescale
        data = self.motion_scaler.inverse_transform(data)
        # Transform to bvh
        bvh = self.motiondata_pipe.inverse_transform([data])[0]
        # Save
        writer = BVHWriter()
        print('writing:' + save_path)
        with open(save_path,'w') as f:
            writer.write(bvh, f, framerate=20)