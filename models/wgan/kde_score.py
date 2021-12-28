import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def calculate_kde_batch(inputs, targets):
    inputs = np.concatenate(inputs, axis=0)
    targets = np.concatenate(targets, axis=0)
    mean, std, se = calculate_kde(inputs, targets)
    return mean, std, se


def calculate_kde(inputs, targets, verbose=0):
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=3)
    grid.fit(inputs)
    kde = grid.best_estimator_
    scores = kde.score_samples(targets)
    mean = np.mean(scores)
    std = np.std(scores)
    se = std / np.sqrt(len(inputs))
    if verbose:
        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
        print(f"mean: {mean}, std: {std}, se: {se}")
    return mean, std, se