import numpy as np
import os
import pandas as pd

def semitone_prosody(audio_filename):

    feature_filename = os.path.splitext(os.path.basename(audio_filename))[0] + '-rms.txt'
    rms_path = './data/takekuchi/source/speech_features'
    feature_path = os.path.join(rms_path, feature_filename)
    
    df = pd.read_table(feature_path)

    power = df['power [dB]'].to_numpy().astype(float)
    pitch = df['f0 after post-proc [semitone]'].to_numpy().astype(float)

    pitch[pitch==-10000] = np.nan

    def my_average(arr, n):
        num_chunk = int(np.ceil(arr.shape[0] / n))
        return np.array([np.nanmean(arr[i * n:(i + 1) * n]) for i in range(num_chunk)])

    power = my_average(power, 5)
    pitch = my_average(pitch, 5)

    # numerical stability
    pitch = np.nan_to_num(pitch, nan=np.finfo(pitch.dtype).eps)

    min_size = min([len(power), len(pitch)])

    power = power[:min_size]
    pitch = pitch[:min_size]

    return np.stack((power, pitch)).T