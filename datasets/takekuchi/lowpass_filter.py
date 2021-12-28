from scipy.signal import butter,filtfilt
import numpy as np


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

def lowpass_filter(data):
    # data (T, n_feature)
    data = np.array(list(map(butter_lowpass_filter, data.T)))
    return data.T