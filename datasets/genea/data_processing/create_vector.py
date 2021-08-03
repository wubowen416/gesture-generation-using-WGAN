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


def create_vector(audiopath:str, bvhpath:str):

    print(audiopath, bvhpath)

    # Speech feature
    input_vec = semitone_prosody(audiopath)

    # Motion feature
    motiondata_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=20,  keep_all=False)),
        ('root', RootTransformer('hip_centric')),
        ('jtsel', JointSelector(['Spine','Spine1','Spine2','Spine3','Neck','Neck1','Head','RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'], include_root=True)),
        ('cnst', ConstantsRemover()),
        ('np', Numpyfier())])

    p = BVHParser()
    data = [p.parse(bvhpath)]
    output_vec = motiondata_pipe.fit_transform(data)[0]

    # Align
    input_vec, output_vec = shorten(input_vec, output_vec)
    return input_vec, output_vec


def create(name:str, audiopaths:list, bvhpaths:list, data_dir:str):

    X, Y = [], []
    for i in range(len(audiopaths)):
        input_vector, output_vector = create_vector(audiopaths[i], bvhpaths[i])
        X.append(input_vector.astype(float))
        Y.append(output_vector.astype(float))

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



if __name__ == "__main__":

    '''
    Converts bvh and wav files into features, slices in equal length intervals and divides the data
    '''

    data_root = '/media/wu/database/GENEA/source'
    bvhpath = os.path.join(data_root, 'bvh')
    audiopath = os.path.join(data_root, 'audio')
    held_out = ['Recording_008']
    processed_dir = 'data/GENEA/processed'

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
    create('train', audiopath_list, bvhpath_list, processed_dir)

    # Held out
    audiopath_list = [
        os.path.join(audiopath, name) + '.wav' for name in held_out]
    bvhpath_list = [
        os.path.join(bvhpath, name) + '.bvh' for name in held_out]
    create('test', audiopath_list, bvhpath_list, processed_dir)
