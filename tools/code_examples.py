import os
import torch
import sys
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np

sys.path.append('.')
from models.wgan.nll_score import calculate_nll


def generate_result_on_test_set(model, data, path):
    output_list, _, motion_list, indexs = model.synthesize_batch(data.get_test_dataset())
    for output, i in zip(output_list, indexs):
        print(os.path.join(f"{path}/motion_{i}.txt"))
        data.save_unity_result(output.cpu().numpy(), os.path.join(f"{path}/motion_{i}.txt"))
    def to_numpy(x):
        return x.cpu().numpy()
    output_list = list(map(to_numpy, output_list))
    motion_list = list(map(to_numpy, motion_list))
    return output_list, motion_list

def evaluate_nll(output_list, motion_list, data):
    output = torch.cat(output_list, dim=0).cpu().numpy()
    motion = torch.cat(motion_list, dim=0).cpu().numpy()
    output = data.motion_scaler.inverse_transform(output)
    motion = data.motion_scaler.inverse_transform(motion)
    nll = calculate_nll(output, motion)
    return nll


def calculate_nll_batch(inputs, targets):
    inputs = np.concatenate(inputs, axis=0)
    targets = np.concatenate(targets, axis=0)
    nll = calculate_nll(inputs, targets)
    return nll


def compute_derivative(data):
    return data[1:] - data[:-1]


def evaluate_gestures(inputs, targets):
    input_vel_list = [compute_derivative(x) for x in inputs]
    input_acc_list = [compute_derivative(x) for x in input_vel_list]
    target_vel_list = [compute_derivative(x) for x in targets]
    target_acc_list = [compute_derivative(x) for x in target_vel_list]
    print('Evaluate nll...')
    nll_rot = calculate_nll_batch(inputs, targets)
    print('nll_rot: ', nll_rot)
    nll_vel = calculate_nll_batch(input_vel_list, target_vel_list)
    print('nll_vel: ', nll_vel)
    nll_acc = calculate_nll_batch(input_acc_list, target_acc_list)
    print('nll_acc: ', nll_acc)
    return nll_rot, nll_vel, nll_acc