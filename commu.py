"""Commu script.
Usage:
    main.py <model> <dataset> <hparams>
"""
import os
from docopt import docopt
from tools.Config import JsonConfig

import datasets
import models
from tools.commu.Client import DataClient, CommuClient
from tools.commu import functions as F
from tools.takekuchi_dataset_tool.rot_to_pos import rot2pos


def send_motion(motion, commu_client):
    # Convert to position
    position = rot2pos(motion)
    # Calculate angle
    rotation_data = F.calculate_rotation_for_shouder(position)
    # Make Commu command
    command = F.CommuCommand(rotation_data)
    # command.to_csv('t.txt')
    send_command = command.to_command()
    # send motion via tcp/ip
    commu_client.sendall(send_command)



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
    
    data = dataset_class(hparams, is_training=False)
    speech_dim, motion_dim = data.get_dims()

    # Load model
    print("Load model...")
    model = model_class(speech_dim, motion_dim, hparams)
    model.build(chkpt_path=hparams.Infer.pre_trained)

    # Commu
    data_client = DataClient()
    commu_client = CommuClient()
    commu_client.reset_pose()
    GENERATED_FLAG = False

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
                sendlen, speech_chunks, seed = F.prepare_generation(model, speech_chunks)
                for i, cond in enumerate(speech_chunks):

                    # First chunk
                    if i == 0:
                        output_motion, seed = F.generate(model, seed, cond, data, interpolate=False)
                        motion = output_motion[:sendlen, :]
                        send_motion(motion, commu_client)

                    # Middle chunks
                    elif i != len(speech_chunks)-1:
                        output_motion, seed = F.generate(model, seed, cond, data, interpolate=True)
                        motion = output_motion[:sendlen, :]
                        send_motion(motion, commu_client)
                    
                    # Last chunk
                    else:
                        output_motion, _ = F.generate(model, seed, cond, data, interpolate=True)
                        motion = output_motion
                        send_motion(motion, commu_client)
            
            # Reset line for next line
            current_line = ""