import numpy as np
import pandas as pd


def cov_matrix(x, is_scaled):
    scaled = x if is_scaled else x - x.mean()
    return 1 / len(x) * (scaled.T @ scaled)


def pca(x, k, is_scaled=True):
    c = cov_matrix(x, is_scaled)
    # TODO: implement selection of k
    u, s, _ = np.linalg.svd(c)
    uk = u[:, :k]
    var = s[:k].sum() / s.sum()
    return (x @ uk), uk.T, var


def pca_inverse(x, uk):
    return x @ uk
