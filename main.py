"""Main script.
Usage:
    main.py <model> <dataset> <hparams>
"""
import os
import datetime
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

    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")
    log_dir = os.path.join(hparams.Dir.log, "log_" + date)
    print("log_dir:" + str(log_dir))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    is_training = hparams.Infer.pre_trained == ""
    
    data = dataset_class(hparams, is_training)
    speech_dim, motion_dim = data.get_dims()

    model = model_class(speech_dim, motion_dim, hparams)
    model.build()

    if is_training:
        model.train(data, log_dir, hparams)
