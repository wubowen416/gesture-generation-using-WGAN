"""Main script.
Usage:
    main.py <model> <dataset> <hparams>
"""
import os
import datetime
from docopt import docopt
from tools.Config import JsonConfig

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

    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")
    log_dir = os.path.join(hparams.Dir.log, "log_" + date)
    # print("log_dir:" + str(log_dir))
    
    
    is_training = hparams.Infer.pre_trained == ""
    
    data = dataset_class(hparams, is_training)
    cond_dim, motion_dim = data.get_dims()

    model = model_class(cond_dim, motion_dim, hparams)
    
    if is_training:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        model.build()
        model.train(data, log_dir, hparams)

    else:
        
        model.build(chkpt_path=hparams.Infer.pre_trained)

        # ---------------------
        # Customize

        # Generate result on test set
        output_list, motion_list = model.synthesize_batch(data.get_test_dataset())
        for i, output in enumerate(output_list):
            data.save_unity_result(output.cpu().numpy(), os.path.join(f"synthesized/motion_{i}.txt"), hip=True)
        # Evaluate KDE
        # import torch
        # from models.wgan.kde_score import calculate_kde
        # output = torch.cat(output_list, dim=0).cpu().numpy()
        # motion = torch.cat(motion_list, dim=0).cpu().numpy()
        # output = data.motion_scaler.inverse_transform(output)
        # motion = data.motion_scaler.inverse_transform(motion)
        # kde_mean, _, kde_se = calculate_kde(output, motion)
        # print(kde_mean, kde_se)
