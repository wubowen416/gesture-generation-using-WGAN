import numpy as np
from scipy import linalg
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


def calculate_frechet_distance(data_pred, data_true, pca=False, dim_pca=15, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    if pca:
        # project to a lower dimension
        data_pred = PCA(n_components=dim_pca, whiten=False).fit_transform(data_pred)
        data_true = PCA(n_components=dim_pca, whiten=False).fit_transform(data_true)

    mu1 = np.mean(data_pred, axis=0)
    sigma1 = np.cov(data_pred, rowvar=False)
    mu2 = np.mean(data_true, axis=0)
    sigma2 = np.cov(data_true, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    frt = (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

    if frt < eps:
        frt = 0.0
    
    return frt


def calculate_kde_score(data_pred: list[np.array], data_true: list[np.array], pca: bool = False, dim_pca: int = 15, bandwidth: list[float] = np.logspace(-1, 1, 20), verbose: bool = True):
    """
    Calculate kde score on true data using model fitted on pred data.
    Args:
        y_pred: numpy array, shape [num_samples, dim]
        y_true: numpy array, shape [num_samples, dim]
        pca: bool, if use PCA to reduce the dimensions of data
        dim_pca: int, to what extent reduce the dimensions
        bandwidth: list of float, search space for fitting KDE
        verboseL: bool, whether display the result
    """

    data_pred = np.concatenate(data_pred, axis=0)
    data_true = np.concatenate(data_true, axis=0)

    if pca:
        # project the 64-dimensional data to a lower dimension
        data_pred = PCA(n_components=dim_pca, whiten=False).fit_transform(data_pred)
        data_true = PCA(n_components=dim_pca, whiten=False).fit_transform(data_true)

    # use grid search cross-validation to optimize the bandwidth
    params = {"bandwidth": bandwidth}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(data_pred)

    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_

    scores = kde.score_samples(data_true)
    ll = np.mean(scores)
    std = np.std(scores)
    se = std / np.sqrt(len(scores))
    if verbose:
        print("KDE best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
        print(f"ll: {ll}, std: {std}, se: {se}")
    return scores, std, se


def calculate_mae(data_pred: np.array, data_true: np.array):
    return np.mean(np.abs(data_pred - data_true))