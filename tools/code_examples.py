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
from models.wgan.kde_score import calculate_kde
from tools.takekuchi_dataset_tool.rot_to_pos import rot2pos


def generate_result_on_test_set(model, data, path):
    output_list, _, motion_list = model.synthesize_batch(data.get_test_dataset())
    for i, output in enumerate(output_list):
        data.save_unity_result(output.cpu().numpy(), os.path.join(f"{path}/motion_{i}.txt"))
    return output_list, motion_list

def evaluate_kde(output_list, motion_list, data):
    output = torch.cat(output_list, dim=0).cpu().numpy()
    motion = torch.cat(motion_list, dim=0).cpu().numpy()
    output = data.motion_scaler.inverse_transform(output)
    motion = data.motion_scaler.inverse_transform(motion)
    kde_mean, _, kde_se = calculate_kde(output, motion)
    return kde_mean, kde_se