"""Commu script.
Usage:
    main.py <model> <dataset> <hparams>
"""
import pickle
import os
from docopt import docopt
from tools.Config import JsonConfig
from scipy.spatial.transform import Rotation as R

import datasets
import models
from tools.commu.Client import DataClient, CommuClient
from tools.commu import functions as F
from tools.takekuchi_dataset_tool.rot_to_pos import rot2pos
from datasets.takekuchi.takekuchi import transform

import numpy as np
import torch
np.random.seed(1)
torch.manual_seed(1)


def send_motion(motion, commu_client, from_numpy=False):
    # Convert to position
    position = rot2pos(motion)
    # Calculate angle
    rotation_data = F.calculate_rotation_for_shouder(position)
    # Make Commu command
    command = F.CommuCommand(rotation_data)
    # command.to_csv('t.txt')
    send_command = command.to_command()
    # send motion via tcp/ip
    if from_numpy:
        input("Press Enter to continue...")
        commu_client.sendall(send_command)
    else:
        commu_client.sendall(send_command)
    return position



if __name__ == "__main__":
    args = docopt(__doc__)
    model_name = args["<model>"]
    hparams_name = args["<hparams>"]
    dataset_name = args["<dataset>"]
    assert model_name in models.model_dict, (
        "`{}` is not supported, use `{}`".format(model_name, models.model_dict.keys()))
    assert dataset_name in datasets.dataset_dict, (
        "`{}` is not supported, use `{}`".format(dataset_name, datasets.dataset_dict.keys()))
    assert os.path.exists(hparams_name), (
        "Failed to find hparams josn `{}`".format(hparams_name))
    hparams = JsonConfig(hparams_name)
    dataset_class = datasets.dataset_dict[dataset_name]
    model_class = models.model_dict[model_name]
    
    assert hparams.Infer.pre_trained != "", "Must provide path for model parameters"
    
    # data = dataset_class(hparams, is_training=False)
    # speech_dim, motion_dim = data.get_dims()

    # Load model
    # print("Load model...")
    # model = model_class(speech_dim, motion_dim, hparams)
    # model.build(chkpt_path=hparams.Infer.pre_trained)

    # Commu
    commu_client = CommuClient()
    commu_client.reset_pose()
    GENERATED_FLAG = False

    if True: # Use numpy file
        path = "synthesized/prosody_va/motion_102.txt" # unity file path
        with open(path, 'r') as f:
            lines = f.readlines()[3:]
        motion = []
        for line in lines:
            motion.append([float(x) for x in line.split(" ")])
        motion = np.array(motion).T

        if motion.shape[-1] == 36:
            with open("data/takekuchi/processed/prosody_hip/Y_dev.p", 'rb') as f:
                Y_dev = pickle.load(f)
            Y_dev = np.concatenate(Y_dev, axis=0)
            hip_mean = np.mean(Y_dev[:, :3], axis=0, keepdims=True)
            hip_mean_transformed = transform(hip_mean).repeat(len(motion), 0)
            motion = np.concatenate([hip_mean_transformed, motion], axis=-1)

        # unity -> numpy
        def yxz_to_zxy(degs):
            r = R.from_euler('YXZ', degs, degrees=True)
            return r.as_euler('ZXY', degrees=True)

        def transform_inverse(vector):
            num_joint = int(vector.shape[1] / 3)
            lines = []
            for frame in vector:
                line = []
                for i in range(num_joint):
                    stepi = i*3
                    x_deg = float(frame[stepi])
                    y_deg = float(frame[stepi+1])
                    z_deg = float(frame[stepi+2])
                    z, x, y = yxz_to_zxy([y_deg, x_deg, z_deg])
                    line.extend([z, x, y])
                lines.append(line)
            return np.array(lines)

        motion = transform_inverse(motion)

        positions = send_motion(motion, commu_client, from_numpy=True)
        # np.save('t.npy', np.concatenate(positions, axis=0))
        assert 0

    else:
        data_client = DataClient()

    

    current_line = ""
    lines = ""

    print("Receiving data...")

    # Stop when at break
    while True:

        # Receive a char every time
        received_char = data_client.receive(size=1)

        # Current line isn't end
        if received_char != "\n":
            current_line += received_char

        # End of current line
        else:
            current_line += "\n"
            lines += current_line

            frame_nunber = int(current_line.split("\t")[0])
            if GENERATED_FLAG and frame_nunber == 0:
                # frame_number = 0 means end of all
                print("Break: end of generation.")
                break
            # End of current sentence
            elif frame_nunber == 1:

                # Generate for current sentence
                if not GENERATED_FLAG:
                    GENERATED_FLAG = True

                speech_chunks = F.data_preprocess(lines, data, hparams)

                # Generate
                positions = []
                sendlen, speech_chunks, seed = F.prepare_generation(model, speech_chunks)
                for i, cond in enumerate(speech_chunks):

                    # First chunk
                    if i == 0:
                        output_motion, seed = F.generate(model, seed, cond, data, interpolate=False)
                        motion = output_motion[:sendlen, :]

                    # Middle chunks
                    elif i != len(speech_chunks)-1:
                        output_motion, seed = F.generate(model, seed, cond, data, interpolate=True)
                        motion = output_motion[:sendlen, :]
                    
                    # Last chunk
                    else:
                        output_motion, _ = F.generate(model, seed, cond, data, interpolate=True)
                        motion = output_motion

                    if i == 0:
                        input("Press Enter to continue...")
                        
                    position = send_motion(motion, commu_client)
                    positions.append(position)

                # np.save('t.npy', np.concatenate(positions, axis=0))
            
            # Reset line for next line
            current_line = ""