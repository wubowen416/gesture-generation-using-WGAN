import librosa
import os
from glob import glob
import soundfile as sf


wav_filepaths = glob(os.path.join('data/takekuchi/source/split/dev/inputs', '*.wav'))

output_path = 'data/takekuchi/source/split/dev/inputs_16k'
os.makedirs(output_path, exist_ok=True)

for wav_filepath in wav_filepaths:
    y, _ = librosa.load(wav_filepath, sr=16000)
    output_filename = os.path.join(output_path, os.path.basename(wav_filepath))
    sf.write(output_filename, y, 16000)