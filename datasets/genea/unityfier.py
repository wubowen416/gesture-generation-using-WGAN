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


class Unityfier:

    def __init__(self):
        pass

    def zxy_to_yxz(self, degs):
        r = R.from_euler('ZXY', degs, degrees=True)
        return r.as_euler('YXZ', degrees=True)

    def line_prepender(self, filename, line):
        with open(filename, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(line.rstrip('\r\n') + '\n' + content)

    def transform(self, vector):
        num_joint = int(vector.shape[1] / 3)
        lines = []
        for frame in vector:
            line = []
            for i in range(num_joint):
                stepi = i*3
                z_deg = float(frame[stepi])
                x_deg = float(frame[stepi+1])
                y_deg = float(frame[stepi+2])

                y, x, z = self.zxy_to_yxz([z_deg, x_deg, y_deg])
                line.extend([x, y, z])
            lines.append(line)
        return np.array(lines)
            
    def filter(self, data):
        # data (T, n_feature)
        data = np.array(list(map(butter_lowpass_filter, data.T)))
        return data.T

    def write_unity(self, vector, save_path):
        unity_lines = self.transform(vector)
        unity_lines = filter(unity_lines)
        np.savetxt(save_path, unity_lines.T)

        prepend_line = f"{vector.shape[0]}\n12\n0.05"
        
        self.line_prepender(save_path, prepend_line)