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
from tools.commu.Client import DataClient, CommuClient, WavClient
from tools.commu import functions as F
from tools.takekuchi_dataset_tool.rot_to_pos import rot2pos
from datasets.takekuchi.takekuchi import transform

import numpy as np
import torch
np.random.seed(1)
torch.manual_seed(1)

from tools.plotting import plot_position_3D


def send_motion(motion, commu_client, wav_client: WavClient = None, from_numpy=False):
    # Convert to position
    position = rot2pos(motion)
    # plot position
    # plot_position_3D(position, 'test.gif')
    # assert 0
    # Calculate angle
    rotation_data = F.calculate_rotation_for_shoulder(position)
    # Make Commu command
    command = F.CommuCommand(rotation_data)
    command.add_joint(joint_idx=0, values=motion[:, 4])
    command.add_joint(joint_idx=1, values=motion[:, 5])
    command.add_joint(joint_idx=7, values=motion[:, 9])
    command.add_joint(joint_idx=8, values=motion[:, 11])
    # command.to_csv('test.txt')
    send_command = command.get_command()
    # send first line to move to first position
    commu_client.send(send_command[0])
    # send motion via tcp/ip
    if from_numpy:
        input("Press Enter to continue...")
        if wav_client != None:
            wav_client.start()
        commu_client.sendall(send_command)
    else:
        if wav_client != None:
            wav_client.start()
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
    # commu_client = None
    commu_client = CommuClient()
    commu_client.reset_pose()
    GENERATED_FLAG = False
    # assert 0

    # Wav client
    wav_client = None
    wav_client = WavClient()
    wav_client.load_wavfile("data/takekuchi/source/split/dev/inputs_16k_sigproc/audio1161.wav")

    if True: # Use unity txt file
        # path = "synthesized/dev_motion/motion_34.txt" # unity file path
        # path = "synthesized/ICMI2021_prosody/motion_34.txt" # unity file path
        path = "synthesized/prosody_va/motion_104.txt" # unity file path

        # get rotation data
        with open(path, 'r') as f:
            lines = f.readlines()[3:]
        motion = []
        for line in lines:
            motion.append([float(x) for x in line.split(" ")])
        motion = np.array(motion).T

        # add root rotation since the model does not output it
        # and unity rotation does not contain it
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

        # convert to command and send
        positions = send_motion(motion, commu_client, wav_client, from_numpy=True)
        # np.savetxt('t.txt', np.concatenate(positions, axis=0))
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