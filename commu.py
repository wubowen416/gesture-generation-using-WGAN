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
    # Connect to server via TCP/IP
    import socket
    import sys
    import torch
    import pandas as pd
    import numpy as np
    from io import StringIO
    from datasets.takekuchi.base_dataset import chunkize

    def alert(msg):
        print(msg)
        sys.exit(1)

    def avg(arr, n):
        num_chunk = int(np.ceil(arr.shape[0] / n))
        return np.array([np.nanmean(arr[i * n:(i + 1) * n])
                       for i in range(num_chunk)])

    def data_preprocess(lines, data):
        # Process data
        # df -> numpy -> chunks
        STRINGDATA = StringIO(HEADER + lines)
        df = pd.read_table(STRINGDATA)
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
        # Scale & chunkize
        scaled_speech_feature = data.speech_scaler.transform(speech_feature)
        return chunkize(scaled_speech_feature, chunklen=hparams.Data.chunklen, stride=hparams.Data.chunklen-hparams.Data.seedlen)

    def prepare_generation(model, speech_chunks):
        model.gen.eval()
        sendlen = model.chunklen - model.seedlen
        speech_chunks = torch.Tensor(speech_chunks).to(model.device)
        seed = torch.zeros(size=(1, model.chunklen, model.output_dim)).to(model.device)
        return sendlen, speech_chunks, seed

    def generate(model, seed, cond, data, interpolate=False):
        cond = cond.unsqueeze(0)
        latent = model.sample_noise(1, device=model.device)
        with torch.no_grad():
            output = model.gen(seed, latent, cond).squeeze(0)
        if interpolate:
            # Interpolation with seed
            for k in range(model.seedlen):
                output[k] = ((model.seedlen - k) / (model.seedlen + 1)) * seed.squeeze(0)[k] + ((k + 1) / (model.seedlen + 1)) * output[k]
        # Save seed motion
        seed[:, :model.seedlen] = output[-model.seedlen:]
        # Rescale & slice & send
        return data.motion_scaler.inverse_transform(output.cpu().numpy()), seed


    ip_addr = "localhost"
    port = 8082

    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print("Connect to server...")
    try:
        tcp_client.connect((ip_addr, port))
    except:
        alert('failed to connect ' + ip_addr)


    ENCODING = "utf-8"
    HEADER = "frame number\t#f0 [Hz]\tperiodicity (0~1)\tpower [dB]\tsonorant power [dB]\thigh freq. power [dB]\tmid freq. power [dB]\tlow freq power [dB]\tf0 after post-proc. [Hz]\tf0 reliability code (0 ~ 13)\ta1-a3 [dB]\tf1f3syn (-1.0~1.0)\th1-a1 [dB]\tbreathpow [dB]\tips (-1.0~1.0)\th1-a3 [dB]\tf0 after post-proc [semitone]\tdelta f0 [semitone]\tvocal fry detection\n"

    GENERATED_FLAG = False
    FLAG_LENGTH = 20

    line = ""
    lines = ""

    print("Receiving data...")

    # Stop when at break
    while True:

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
            if GENERATED_FLAG and frame_nunber == 0:
                # frame_number = 0 means end of all
                print("Break: end of generation.")
                break
            # End of current sentence
            elif frame_nunber == 1:

                # Generate for current sentence
                if not GENERATED_FLAG:
                    GENERATED_FLAG = True

                speech_chunks = data_preprocess(lines, data)

                # Generate
                sendlen, speech_chunks, seed = prepare_generation(model, speech_chunks)
                for i, cond in enumerate(speech_chunks):

                    # First chunk
                    if i == 0:
                        output_motion, seed = generate(model, seed, cond, data, interpolate=False)
                        motion = output_motion[:sendlen, :]
                        # send motion via tcp/ip
                        print("send to server first chunk.")

                    # Middle chunks
                    elif i != len(speech_chunks)-1:
                        output_motion, seed = generate(model, seed, cond, data, interpolate=True)
                        motion = output_motion[:sendlen, :]
                        # send motion via tcp/ip
                        print("send to server middle chunk.")
                    
                    # Last chunk
                    else:
                        output_motion, seed = generate(model, seed, cond, data, interpolate=True)
                        motion = output_motion
                        # send motion via tcp/ip
                        print("send to server last chunk.")
            
            # Reset line for next line
            line = ""
