import os
import numpy as np
from sklearn.pipeline import Pipeline
import pickle
from sklearn.preprocessing import StandardScaler
import joblib as jl
from prosodic_feature import semitone_prosody
from pymo.parsers import BVHParser
from pymo.preprocessing import *


def shorten(arr1, arr2):
    min_len = min(len(arr1), len(arr2))
    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]
    return arr1, arr2


def create_vector(audiopaths:list, bvhpaths:list, motiondata_pipe=None):

    p = BVHParser()
    audio_vecs, bvhs = [], []

    for audiopath, bvhpath in zip(audiopaths, bvhpaths):

        print(audiopath, bvhpath)

        # Speech feature
        audio_vecs.append(semitone_prosody(audiopath))

        # Bvh
        bvhs.append(p.parse(bvhpath))

    # Motion feature
    if not motiondata_pipe:
        motiondata_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=20,  keep_all=False)),
            ('root', RootTransformer('hip_centric')),
            ('jtsel', JointSelector(['Spine','Spine1','Spine2','Spine3','Neck','Neck1','Head','RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'], include_root=True)),
            ('cnst', ConstantsRemover()),
            ('np', Numpyfier())])
        motion_vecs = motiondata_pipe.fit_transform(bvhs)
    else:
        motion_vecs = motiondata_pipe.transform(bvhs)

    # Align
    input_vecs, output_vecs = [], []
    for audio_vec, motion_vec in zip(audio_vecs, motion_vecs):
        audio_vec, motion_vec = shorten(audio_vec, motion_vec)
        input_vecs.append(audio_vec)
        output_vecs.append(motion_vec)

    return input_vecs, output_vecs, motiondata_pipe


def create(name:str, audiopaths:list, bvhpaths:list, data_dir:str, motiondata_pipe=None):

    os.makedirs(data_dir, exist_ok=True)

    if name == "train":
        input_vectors, output_vectors, motiondata_pipe = create_vector(audiopaths, bvhpaths)
        input_scaler = StandardScaler().fit(np.concatenate(input_vectors, axis=0))
        output_scaler = StandardScaler().fit(np.concatenate(output_vectors, axis=0))
        jl.dump(input_scaler, os.path.join(data_dir, "speech_scaler.jl"))
        jl.dump(output_scaler, os.path.join(data_dir, "motion_scaler.jl"))
        jl.dump(motiondata_pipe, os.path.join(data_dir, "motiondata_pipe.jl"))

    elif name == "test":
        input_vectors, output_vectors, _ = create_vector(audiopaths, bvhpaths, motiondata_pipe)

    x_file_name = os.path.join(data_dir, 'X_' + str(name) + '.p')
    y_file_name = os.path.join(data_dir, 'Y_' + str(name) + '.p')
    with open(x_file_name, 'wb+') as f:
        pickle.dump(input_vectors, f)
    with open(y_file_name, 'wb+') as f:
        pickle.dump(output_vectors, f)
    return motiondata_pipe


if __name__ == "__main__":

    '''
    Converts bvh and wav files into features, slices in equal length intervals and divides the data
    '''

    data_root = '/media/wu/database/GENEA/source'
    bvhpath = os.path.join(data_root, 'bvh')
    audiopath = os.path.join(data_root, 'audio')
    held_out = ['Recording_008']
    processed_dir = './data/genea/processed'

    # Train
    # Get train file name
    filename_list = ["Recording_00{}".format(str(k)) for k in range(1, 10)] + \
        ["Recording_0{}".format(str(k)) for k in range(10, 24)]
    for held_name in held_out:
        filename_list.remove(held_name)

    # get audio file path
    audiopath_list = [
        os.path.join(audiopath, name) + '.wav' for name in filename_list]
    bvhpath_list = [
        os.path.join(bvhpath, name) + '.bvh' for name in filename_list]

    # Process data
    motiondata_pipe = create('train', audiopath_list, bvhpath_list, processed_dir)

    # Held out
    audiopath_list = [
        os.path.join(audiopath, name) + '.wav' for name in held_out]
    bvhpath_list = [
        os.path.join(bvhpath, name) + '.bvh' for name in held_out]
    create('test', audiopath_list, bvhpath_list, processed_dir, motiondata_pipe)
