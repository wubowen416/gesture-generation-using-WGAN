import numpy as np
import os
import pandas as pd

import scipy.io.wavfile as wav
from python_speech_features import mfcc

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


def average(arr, n):
    """ Replace every "n" values by their average
    Args:
        arr: input array
        n:   number of elements to average on
    Returns:
        resulting array
    """
    end = n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


def calculate_mfcc(audio_filename):
    """
    Calculate MFCC features for the audio in a given file
    Args:
        audio_filename: file name of the audio
    Returns:
        feature_vectors: MFCC feature vector for the given audio file
    """
    MFCC_INPUTS = 26

    fs, audio = wav.read(audio_filename)

    # Make stereo audio being mono
    if len(audio.shape) == 2:
        audio = (audio[:, 0] + audio[:, 1]) / 2

    # Calculate MFCC feature with the window frame it was designed for
    input_vectors = mfcc(audio, winlen=0.02, winstep=0.01, samplerate=fs, numcep=MFCC_INPUTS)

    input_vectors = [average(input_vectors[:, i], 5) for i in range(MFCC_INPUTS)]

    feature_vectors = np.transpose(input_vectors)

    return feature_vectors