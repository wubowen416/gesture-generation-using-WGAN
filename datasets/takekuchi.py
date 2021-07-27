import numpy as np
import os
import pickle
import joblib as jl
from sklearn.model_selection import train_test_split
from .base_dataset import TrainDataset, TestDataset

### unity data
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.signal import butter,filtfilt

def butter_lowpass_filter(data):
    # Params ajusted for this dataset
    # Filter requirements.
    T = len(data) / 20         # Sample Period
    fs = 20.0       # sample rate, Hz
    cutoff = 2.5     
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    n = int(T * fs) # total number of samples
    
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def zxy_to_yxz(degs):
    r = R.from_euler('ZXY', degs, degrees=True)
    return r.as_euler('YXZ', degrees=True)

def transform(vector):
    num_joint = int(vector.shape[1] / 3)
    lines = []
    for frame in vector:
        line = []
        for i in range(num_joint):
            stepi = i*3
            z_deg = float(frame[stepi])
            x_deg = float(frame[stepi+1])
            y_deg = float(frame[stepi+2])

            y, x, z = zxy_to_yxz([z_deg, x_deg, y_deg])
            line.extend([x, y, z])
        lines.append(line)
    return np.array(lines)

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
        
def filter(data):
    # data (T, n_feature)
    data = np.array(list(map(butter_lowpass_filter, data.T)))
    return data.T

def write_unity(vector, save_path):
    unity_lines = transform(vector)
    unity_lines = filter(unity_lines)
    np.savetxt(save_path, unity_lines.T)

    prepend_line = f"{vector.shape[0]}\n12\n0.05"
    
    line_prepender(save_path, prepend_line)
###

class TakekuchiDataset:

    def __init__(self, hparams, is_training):

        data_dir = hparams.Dir.processed
        
        # Load scalers
        self.speech_scaler = jl.load(os.path.join(data_dir, 'speech_scaler.jl'))
        self.motion_scaler = jl.load(os.path.join(data_dir, 'motion_scaler.jl'))

        if is_training:

            # Load data and scale
            with open(os.path.join(data_dir, "X_train.p"), 'rb') as f:
                train_input = pickle.load(f)
            with open(os.path.join(data_dir, "Y_train.p"), 'rb') as f:
                train_output = pickle.load(f)

            train_input = list(map(self.speech_scaler.transform, train_input))
            train_output = list(map(self.motion_scaler.transform, train_output))

            # Split validation 
            train_input, val_input, train_output, val_output = train_test_split(
                train_input, train_output, test_size=hparams.Data.valid_ratio, random_state=2021)

            # Create pytorch dataset
            self.train_dataset = TrainDataset(train_input, train_output, hparams.Data.chunklen, hparams.Data.seedlen, stride=1)
            self.val_dataset = TrainDataset(val_input, val_output,  hparams.Data.chunklen, hparams.Data.seedlen, stride=1)

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
            with open(os.path.join(data_dir, "X_test.p"), 'rb') as f:
                test_input = pickle.load(f)
            with open(os.path.join(data_dir, "Y_test.p"), 'rb') as f:
                test_output = pickle.load(f)

            test_input = list(map(self.speech_scaler.transform, test_input))
            test_output = list(map(self.motion_scaler.transform, test_output))
            self.test_dataset = TestDataset(test_input, test_output)

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
        return self.speech_dim, self.motion_scaler

    def save_unity_result(self, data, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
        # rescale
        data = self.motion_scaler.inverse_transform(data)
        # Unitize
        unity_lines = transform(data)
        unity_lines = filter(unity_lines)
        # Save
        np.savetxt(save_path, unity_lines.T)
        prepend_line = f"{data.shape[0]}\n12\n0.05"
        line_prepender(save_path, prepend_line)