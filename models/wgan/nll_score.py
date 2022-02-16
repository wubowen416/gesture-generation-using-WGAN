import math
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


# def calculate_kde_batch(inputs, targets):
#     inputs = np.concatenate(inputs, axis=0)
#     targets = np.concatenate(targets, axis=0)
#     mean, std, se = calculate_kde(inputs, targets)
#     return mean, std, se


# def calculate_kde(inputs, targets, verbose=0):
#     params = {'bandwidth': np.logspace(-1, 1, 20)}
#     grid = GridSearchCV(KernelDensity(), params, cv=3)
#     grid.fit(inputs)
#     kde = grid.best_estimator_
#     scores = kde.score_samples(targets)
#     mean = np.mean(scores)
#     std = np.std(scores)
#     se = std / np.sqrt(len(inputs))
#     if verbose:
#         print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
#         print(f"mean: {mean}, std: {std}, se: {se}")
#     return mean, std, se

def calculate_nll_batch(inputs, targets):
    inputs = np.concatenate(inputs, axis=0)
    targets = np.concatenate(targets, axis=0)
    nll = calculate_nll(inputs, targets)
    return nll


def calculate_nll(inputs, targets):
    """
    Calculate likelihood of targets based on inputs. bits per dim.

    :param inputs: an [T, D] numpy array.
    :param targets: and [T, D] numpy array.
    """
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=3)
    grid.fit(inputs)
    kde = grid.best_estimator_
    scores = kde.score(targets) # Total log-likelihood of the data in X. summation. in nats.
    nll = -scores / inputs.shape[0] / inputs.shape[1] / math.log(2)
    return nll