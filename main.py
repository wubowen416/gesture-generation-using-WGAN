"""Main script.
Usage:
    main.py <model> <dataset> <hparams>
"""
import os
import datetime
from docopt import docopt
from tools.JsonConfig import JsonConfig
from tools import code_examples as exp

import datasets
import models

import torch
import numpy as np


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
    print("log_dir:" + str(log_dir))
    
    is_training = hparams.Infer.pre_trained == ""

    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    
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
        # Some samples can be found in ./tools/code_examples.py
        # ---------------------

        # Generate result on test set
        os.makedirs('synthesized', exist_ok=True)
        output_list, motion_list = exp.generate_result_on_test_set(model, data, f'synthesized/{hparams.run_name}')

        # Evaluate KDE
        # kde_mean, kde_se = exp.evaluate_kde(output_list, motion_list, data)
        # with open('kde_result.txt', 'w') as f:
        #     print(f"kde_mean: {kde_mean}\nkde_se: {kde_se}", file=f)
