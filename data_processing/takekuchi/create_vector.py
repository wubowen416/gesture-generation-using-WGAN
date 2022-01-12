import pandas as pd
import tqdm
import pickle
import argparse
import os
import numpy as np
import joblib as jl
from sklearn.preprocessing import StandardScaler

from bvh_to_rot import vectorize_bvh_to_rotation
from audio_feature import semitone_prosody, calculate_mfcc
from text_feature import frame_encoding


FEATURE_TYPE = "prosody_text"


def shorten(arr1, arr2):
    min_len = min(len(arr1), len(arr2))
    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]
    return arr1, arr2


def create_vectors(audio_filename, gesture_filename):

    # Step 1: speech features
    if FEATURE_TYPE == "prosody":
        input_vectors = semitone_prosody(audio_filename)
    elif FEATURE_TYPE == "mfcc":
        input_vectors = calculate_mfcc(audio_filename)
    elif FEATURE_TYPE == "mfcc_prosody":
        prosody_vectors = semitone_prosody(audio_filename)
        mfcc_vectors = calculate_mfcc(audio_filename)
        prosody_vectors, mfcc_vectors = shorten(prosody_vectors, mfcc_vectors)
        input_vectors = np.concatenate([prosody_vectors, mfcc_vectors], axis=-1)
    elif FEATURE_TYPE == "prosody_text":
        prosody_vectors = semitone_prosody(audio_filename)
        text_vectors = frame_encoding(audio_filename)
        if text_vectors is None:
            return None, None
        if text_vectors.shape[0] < prosody_vectors.shape[0]:
            text_vectors = np.concatenate([np.zeros((prosody_vectors.shape[0] - text_vectors.shape[0], text_vectors.shape[-1])), text_vectors], axis=0)
        prosody_vectors, text_vectors = shorten(prosody_vectors, text_vectors)
        input_vectors = np.concatenate([prosody_vectors, text_vectors], axis=-1)

    # Step 2: Vectorize BVH
    output_vectors = vectorize_bvh_to_rotation(gesture_filename)

    # Check
    # print(input_vectors.shape)
    # print(output_vectors.shape)
    # assert 0

    # Step 3: Align vector length
    input_vectors, output_vectors = shorten(input_vectors, output_vectors)

    return input_vectors, output_vectors


def create(name, dataset_dir, data_dir):

    df_path = os.path.join(dataset_dir, 'gg-' + str(name) + '.csv')

    df = pd.read_csv(df_path)
    X, Y = [], []

    for i in tqdm.tqdm(range(len(df)), ascii=True):
        input_vectors, output_vectors = create_vectors(df['wav_filename'][i], df['bvh_filename'][i])

        if input_vectors is None:
            print(f"{df['wav_filename'][i]} processing failed. Skip")
            continue

        X.append(input_vectors.astype(float))
        Y.append(output_vectors.astype(float))

    os.makedirs(data_dir, exist_ok=True)
    x_file_name = os.path.join(data_dir, 'X_' + str(name) + '.p')
    y_file_name = os.path.join(data_dir, 'Y_' + str(name) + '.p')
    with open(x_file_name, 'wb+') as f:
        pickle.dump(X, f)
    with open(y_file_name, 'wb+') as f:
        pickle.dump(Y, f)

    if name == "train":
        input_scaler = StandardScaler().fit(np.concatenate(X, axis=0))
        output_scaler = StandardScaler().fit(np.concatenate(Y, axis=0))
        jl.dump(input_scaler, os.path.join(data_dir, "speech_scaler.jl"))
        jl.dump(output_scaler, os.path.join(data_dir, "motion_scaler.jl"))


def parse_arg():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset-dir', default='/media/wu/database/speech-to-gesture-takekuchi-2017/split',
    #                     help="Specify dataset dir (containing gg-train, gg-dev, gg-test)")
    parser.add_argument('--dataset-dir', default='./data/takekuchi/source/split',
                        help="Specify dataset dir (containing gg-train, gg-dev, gg-test)")
    parser.add_argument('--data-dir', default=f'./data/takekuchi/processed/{FEATURE_TYPE}_flt', # flt represents filtered
                        help="Specify processed data save dir")
    return parser.parse_args()


if __name__ == "__main__":

    # Specify data dir
    args = parse_arg()

    create('train', args.dataset_dir, args.data_dir)
    create('dev', args.dataset_dir, args.data_dir)
    create('test', args.dataset_dir, args.data_dir)

