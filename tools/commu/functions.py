import torch
import warnings
import os
import pandas as pd
import numpy as np
from io import StringIO
from tools.takekuchi_dataset_tool.functions import chunkize


def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)

def avg(arr, n):
    num_chunk = int(np.ceil(arr.shape[0] / n))
    with warnings.catch_warnings():
        # RuntimeWarning: Mean of empty slice.
        # Consecutive nan value in pitch, as people sometimes is not speaking
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.array([np.nanmean(arr[i * n:(i + 1) * n])
                       for i in range(num_chunk)])

def normalize(arr, axis=-1, order=2):
    l2 = np.linalg.norm(arr, ord=order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return arr / l2

def data_preprocess(lines, data, hparams):
    # Process data
    HEADER = "frame number\t#f0 [Hz]\tperiodicity (0~1)\tpower [dB]\tsonorant power [dB]\thigh freq. power [dB]\tmid freq. power [dB]\tlow freq power [dB]\tf0 after post-proc. [Hz]\tf0 reliability code (0 ~ 13)\ta1-a3 [dB]\tf1f3syn (-1.0~1.0)\th1-a1 [dB]\tbreathpow [dB]\tips (-1.0~1.0)\th1-a3 [dB]\tf0 after post-proc [semitone]\tdelta f0 [semitone]\tvocal fry detection\n"
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

def calculate_rotation_for_shouder(position):
    """36 dim position data from hip"""
    position = position.reshape(-1, 12, 3)
    # Swap z, x, y to x, y, z
    position[:, :, [0, 1, 2]] = position[:, :, [2, 0, 1]]

    # Left
    shoulder_index = 6
    hand_index = 8
    shouder_position = position[:, shoulder_index, :]
    hand_position = position[:, hand_index, :]
    direction_vector = normalize(hand_position - shouder_position)

    xs, ys, zs = direction_vector[:, 0], direction_vector[:, 1], direction_vector[:, 2]

    data = {"left": {}, "right": {}}

    alphas = []
    betas = []
    for x, y, z in zip(xs, ys, zs):
        if z > 0:
            if x > 0:
                alpha = np.rad2deg(np.arctan(x/z))
            else:
                alpha = 270 + np.rad2deg(np.arctan(z/np.abs(x)))
            if y > 0:
                beta = np.rad2deg(np.arctan(y/z))
            else:
                beta = 270 + np.rad2deg(np.arctan(y/np.abs(z)))
        elif z < 0:
            if x < 0:
                alpha = 180 + np.rad2deg(np.arctan(np.abs(x)/np.abs(z)))
            else:
                alpha = 90 + np.rad2deg(np.arctan(np.abs(z)/x))
            if y < 0:
                beta = 180 + np.rad2deg(np.arctan(np.abs(y)/np.abs(z)))
            else:
                beta = 90 + np.rad2deg(np.arctan(np.abs(z)/y))
        alphas.append(alpha)
        betas.append(beta)

    data["left"]["alpha"] = np.array(alphas)
    data["left"]["beta"] = np.array(betas)

    # Right
    shoulder_index = 9
    hand_index = 11
    shouder_position = position[:, shoulder_index, :]
    hand_position = position[:, hand_index, :]
    direction_vector = normalize(hand_position - shouder_position)

    xs, ys, zs = direction_vector[:, 0], direction_vector[:, 1], direction_vector[:, 2]

    alphas = []
    betas = []
    for x, y, z in zip(xs, ys, zs):
        if z > 0:
            if x > 0:
                alpha = np.rad2deg(np.arctan(x/z))
            else:
                alpha = 270 + np.rad2deg(np.arctan(z/np.abs(x)))
            if y > 0:
                beta = np.rad2deg(np.arctan(y/z))
            else:
                beta = 270 + np.rad2deg(np.arctan(y/np.abs(z)))
        elif z < 0:
            if x < 0:
                alpha = 180 + np.rad2deg(np.arctan(np.abs(x)/np.abs(z)))
            else:
                alpha = 90 + np.rad2deg(np.arctan(np.abs(z)/x))
            if y < 0:
                beta = 180 + np.rad2deg(np.arctan(np.abs(y)/np.abs(z)))
            else:
                beta = 90 + np.rad2deg(np.arctan(np.abs(z)/y))
        alphas.append(alpha)
        betas.append(beta)

    data["right"]["alpha"] = np.array(alphas)
    data["right"]["beta"] = np.array(betas)
    return data


def rotation2commucommand(data):

    # tate

    left_tate_euler = data["left"]["alpha"]
    right_tate_euler = data["right"]["alpha"]

    left_tate_euler[left_tate_euler > 180] = 180
    left_tate_euler[right_tate_euler < 0] = 0
    right_tate_euler[right_tate_euler > 180] = 180
    right_tate_euler[right_tate_euler < 0] = 0

    left_tate_euler[left_tate_euler < 90] = 90 - left_tate_euler[left_tate_euler < 90]
    left_tate_euler[left_tate_euler > 90] = -(left_tate_euler[left_tate_euler > 90] - 90)
    right_tate_euler[right_tate_euler < 90] = 90 - right_tate_euler[right_tate_euler < 90]
    right_tate_euler[right_tate_euler > 90] = -(right_tate_euler[right_tate_euler > 90] - 90)

    
    # yoko
    # left
    left_yoko_euler = data["left"]["beta"]
    left_yoko_euler[left_yoko_euler > 180] = 180
    left_yoko_euler[left_yoko_euler < 0] = 0
    left_yoko_euler[left_yoko_euler < 90] = 90 - left_yoko_euler[left_yoko_euler < 90]
    left_yoko_euler[left_yoko_euler > 90] = - (left_yoko_euler[left_yoko_euler > 90] - 90)
    left_yoko_euler[left_yoko_euler < -45] = -45
    left_yoko_euler[left_yoko_euler > 25] = 25

    # right
    right_yoko_euler = data["right"]["beta"]
    right_yoko_euler[right_yoko_euler < 180] = 180
    right_yoko_euler[right_yoko_euler < 270] = - (270 - right_yoko_euler[right_yoko_euler < 270])
    right_yoko_euler[right_yoko_euler > 270] = right_yoko_euler[right_yoko_euler > 270] - 270
    right_yoko_euler[right_yoko_euler < -45] = -45
    right_yoko_euler[right_yoko_euler > 25] = 25


    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use("agg")
    # plt.plot(right_yoko_euler)
    # plt.savefig("t.jpg")

    # Make commu command
    # Write command
    MAX_SPEED = 50
    DENOMINATOR = 5
    SEND_FPS = 10
    length = len(left_tate_euler)
    sheet = np.full(shape=(14, length), fill_value=-10000, dtype=int)

    # lines = []

    for i in range(length):

        # line = self.command

        if i == 0:
            speed_2 = MAX_SPEED
            speed_3 = MAX_SPEED
            speed_4 = MAX_SPEED
            speed_5 = MAX_SPEED

        else:
            speed_2 = int(np.abs((left_tate_euler[i] - left_tate_euler[i-1]) / (1 / SEND_FPS))) // DENOMINATOR
            speed_3 = int(np.abs((left_yoko_euler[i] - left_yoko_euler[i-1]) / (1 / SEND_FPS))) // DENOMINATOR
            speed_4 = int(np.abs((right_tate_euler[i] - right_tate_euler[i-1]) / (1 / SEND_FPS))) // DENOMINATOR
            speed_5 = int(np.abs((right_yoko_euler[i] - right_yoko_euler[i-1]) / (1 / SEND_FPS))) // DENOMINATOR

        if speed_2 != 0:
            # line += f" 2 {left_tate_euler[i]} {speed_2}"
            sheet[2, i] = left_tate_euler[i]

        if speed_3 != 0:
            # line += f" 3 {left_yoko_euler[i]} {speed_3}"
            sheet[3, i] = left_yoko_euler[i]

        if speed_4 != 0:
            # line += f" 4 {right_tate_euler[i]} {speed_4}"
            sheet[4, i] = right_tate_euler[i]

        if speed_5 != 0:
            # line += f" 5 {right_yoko_euler[i]} {speed_5}"
            sheet[5, i] = right_yoko_euler[i]

        # if speed_2 == 0 and speed_3 == 0 and speed_4 == 0 and speed_5 == 0:
            # line = "skip"
            # pass

        if speed_2 == 0:
            if i == 0:
                sheet[2, i] = 80
            else:
                sheet[2, i-1] = left_tate_euler[i-1]
        if speed_3 == 0:
            if i == 0:
                sheet[3, i] = -4
            else:
                sheet[3, i-1] = left_yoko_euler[i-1]
        if speed_4 == 0:
            if i == 0:
                sheet[4, i] = -80
            else:
                sheet[4, i-1] = right_tate_euler[i-1]
        if speed_5 == 0:
            if i == 0:
                sheet[4, i] = 4
            else:
                sheet[5, i-1] = right_yoko_euler[i-1]

    sheet = sheet.T

    # Write to file
    output_path = "t.txt"
    np.savetxt(output_path, sheet.astype(int), delimiter="\t", fmt='%i')

    # Prepend header
    line_0 = "comment:2nd row is default values: 3rd is axis number: 4th are motion inverval, motion steps, motion axes"
    line_1 = "0\t0\t80\t-4\t-80\t4\t0\t0\t0\t0\t0\t0\t-5\t0"
    line_2 = "0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13"
    line_3 = "10\t{}\t14".format(len(sheet))
    prepend_line(output_path, line_3)
    prepend_line(output_path, line_2)
    prepend_line(output_path, line_1)
    prepend_line(output_path, line_0)

    # Append last line
    # Open a file with access mode 'a'
    last_line = "*" + " 111" * 14
    with open(output_path, "a") as f:
    # Append 'hello' at the end of file
        f.write(last_line)










        






    

    

