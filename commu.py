"""Commu script.
Usage:
    main.py <model> <dataset> <hparams>
"""
import os
from docopt import docopt
from tools.config import JsonConfig

import datasets
import models


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
    # 1. Receive data from server
    #   1.1 Read data as panda dataframe via "layer2"
    #       1.1.1 Wait for start flag
    #       1.1.2 Reading data while end flag appears
    #       1.1.3 Save data via "layer2" protocol
    #   1.2 Convert data to numpy array

    # 2. Generate gesture data
    #   2.1 Chunkize
    #   2.2 Scale (Commu self-scaling)
    #   2.3 Generate using model
    #       2.3.1 Generate for each chunk (loop)
    #           2.3.1.1 Convert gesture data to commu command
    #           2.3.1.2 Send data to server

    # 3. Convert gesture data to commu command
    #   3.1 Position to rotation
    #   3.2 Commu command
    # 4. Send data to server

    
    # Connect to server via TCP/IP
    import socket
    import sys
    import pandas as pd
    import numpy as np
    from io import StringIO

    def alert(msg):
        print(msg)
        sys.exit(1)

    def avg(arr, n):
        num_chunk = int(np.ceil(arr.shape[0] / n))
        return np.array([np.nanmean(arr[i * n:(i + 1) * n])
                       for i in range(num_chunk)])

    ip_addr = "localhost"
    port = 8081

    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print("Connect to server...")
    try:
        tcp_client.connect((ip_addr, port))
    except:
        alert('failed to connect ' + ip_addr)


    ENCODING = "utf-8"
    HEADER = "frame number\t#f0 [Hz]\tperiodicity (0~1)\tpower [dB]\tsonorant power [dB]\thigh freq. power [dB]\tmid freq. power [dB]\tlow freq power [dB]\tf0 after post-proc. [Hz]\tf0 reliability code (0 ~ 13)\ta1-a3 [dB]\tf1f3syn (-1.0~1.0)\th1-a1 [dB]\tbreathpow [dB]\tips (-1.0~1.0)\th1-a3 [dB]\tf0 after post-proc [semitone]\tdelta f0 [semitone]\tvocal fry detection\n"

    END_ALL_STR = "end_all"
    START_STR = "start"
    FLAG_LENGTH = 20

    END_ALL_BUFFER = END_ALL_STR.ljust(FLAG_LENGTH)
    START_BUFFER = START_STR.ljust(FLAG_LENGTH)

    line = ""
    lines = ""

    print("Receiving data...")

    # Stop when received END_ALL flag
    while line != END_ALL_BUFFER:

        # Receive a char every time
        recv_char = tcp_client.recv(1).decode(ENCODING)

        # Current line isn't end
        if recv_char != "\n":
            line += recv_char

        # End of current line
        else:
            line += "\n"
            lines += line

            frame_nunber = int(line.split("\t")[0])
            # ENd of current sentence
            if frame_nunber == 1:

                # Generate for current sentence

                STRINGDATA = StringIO(HEADER + lines)
                df = pd.read_table(STRINGDATA)

                print(df)

                # Process data
                # df -> numpy -> chunks
                power = df['power [dB]'].to_numpy().astype(float)
                pitch = df['f0 after post-proc [semitone]'].to_numpy().astype(float)
                pitch[pitch == -10000] = np.nan
                power = avg(power, 5)
                pitch = avg(pitch, 5)
                # numerical stability
                pitch = np.nan_to_num(pitch, nan=np.finfo(pitch.dtype).eps)
                min_size = min([len(power), len(pitch)])
                power = power[:min_size]
                pitch = pitch[:min_size]
                speech_feature = np.stack((power, pitch)).T



            
            # Reset line for next line
            line = ""
