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
from datasets.takekuchi_ext.base_dataset_ext import velocity_sum, avg_hand_amplitude


def generate_result_on_test_set(model, data):
    output_list, motion_list = model.synthesize_batch(data.get_test_dataset())
    for i, output in enumerate(output_list):
        data.save_unity_result(output.cpu().numpy(), os.path.join(f"synthesized/motion_{i}.txt"), hip=True)
    return output_list, motion_list

def evaluate_kde(output_list, motion_list, data):
    output = torch.cat(output_list, dim=0).cpu().numpy()
    motion = torch.cat(motion_list, dim=0).cpu().numpy()
    output = data.motion_scaler.inverse_transform(output)
    motion = data.motion_scaler.inverse_transform(motion)
    kde_mean, _, kde_se = calculate_kde(output, motion)
    return kde_mean, kde_se

def category_result_verification(output_list, data, num_categories=3):
    
    value_list = []
    category_list = []
    attribute_list = []
    
    for cate in range(num_categories):
        outputs = output_list[cate::num_categories]
        # Move to cpu
        for i in range(len(outputs)):
            outputs[i] = outputs[i].cpu().numpy()
        eulers = list(map(data.motion_scaler.inverse_transform, outputs))
        positions = list(map(rot2pos, eulers))
        vels = list(map(velocity_sum, positions))
        amps = list(map(avg_hand_amplitude, positions))

        value_list += vels
        attribute_list += ["vel"] * len(vels)
        category_list += [str(cate)] * len(vels)

        value_list += amps
        attribute_list += ["amp"] * len(amps)
        category_list += [str(cate)] * len(amps)

    data_dict = {
        "value": value_list,
        "attribute": attribute_list,
        "category": category_list
    }

    df = pd.DataFrame(data_dict)

    # plot vel
    df = df[df["attribute"] == "vel"]
    # plot amp
    # df = df[df["attribute"] == "amp"]

    fig = plt.figure(dpi=150)
    sns.boxplot(x="category", y="value", data=df)
    plt.savefig("figure.jpg")