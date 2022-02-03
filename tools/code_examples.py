import os
import torch
import sys
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd

sys.path.append('.')
from models.wgan.nll_score import calculate_nll
from tools.takekuchi_dataset_tool.rot_to_pos import rot2pos


def generate_result_on_test_set(model, data, path):
    output_list, _, motion_list, indexs = model.synthesize_batch(data.get_test_dataset())
    for output, i in zip(output_list, indexs):
        print(os.path.join(f"{path}/motion_{i}.txt"))
        data.save_unity_result(output.cpu().numpy(), os.path.join(f"{path}/motion_{i}.txt"))
    return output_list, motion_list

def evaluate_nll(output_list, motion_list, data):
    output = torch.cat(output_list, dim=0).cpu().numpy()
    motion = torch.cat(motion_list, dim=0).cpu().numpy()
    output = data.motion_scaler.inverse_transform(output)
    motion = data.motion_scaler.inverse_transform(motion)
    nll = calculate_nll(output, motion)
    return nll