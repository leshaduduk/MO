import math

import pandas as pd
import numpy as np


def create_design_matrix(x, normalize=True):
    """
    Obsolete. Use sklearn.preprocessing
    """
    ones = pd.Series(1, index=np.arange(len(x)))
    new_x, means, widths = _normalize(x) if normalize else (x, 0, 1)
    means = np.insert(means, 0, 0)
    widths = np.insert(widths, 0, 1)
    result = pd.concat(
        [ones, new_x], axis="columns", names=(["dummy"].extend(x.columns))
    )
    return result, means, widths


def denormalize_known(x, means, widths, index=None):
    """
    Obsolete. Use sklearn.preprocessing
    """
    if index is None:
        return widths * x + means
    return widths[index] * x + means[index]


def gradient_descent(theta, x, y, alpha, reg_param=0):
    gradient = lambda _theta: logistic_loss_gradient(_theta, x, y, reg_param)
    for current_theta in iterate_theta(gradient, theta, alpha):
        loss = logistic_loss(current_theta, x, y, reg_param)
        yield current_theta, loss


def iterate_theta(gradient, theta, alpha):
    while True:
        yield theta
        theta = theta - alpha * gradient(theta)


def logistic_cost(theta, x, y):
    prediction = logistic_hypothesis(x, theta)
    return -np.log(y * prediction + (1 - y) * (1 - prediction))


def logistic_hypothesis(x, theta):
    return sigmoid(np.dot(x, theta))


def logistic_loss(theta, x, y, reg_param=0):
    cost_sum = logistic_cost(theta, x, y).sum()
    reg_sum = reg_param / 2 * np.power(theta[1:], 2).sum()
    return 1 / len(x) * (cost_sum + reg_sum)


def logistic_loss_gradient(theta, x, y, reg_param=0):
    cost_sum = np.dot(x.transpose(), logistic_hypothesis(x, theta) - y)
    reg_sum = np.multiply(reg_param, theta)
    reg_sum[0] = 0
    return 1 / len(x) * (cost_sum + reg_sum)


def normalize_known(x, means, widths, index=None):
    """
    Obsolete. Use sklearn.preprocessing
    """
    if index is None:
        return (x - means) / widths
    return (x - means[index]) / widths[index]


def run_descent(alpha, tolerance, theta_0, x, y, reg_param=0):
    prev_loss = None
    theta = None
    progress = []
    for current_theta, loss in gradient_descent(theta_0, x, y, alpha, reg_param):
        progress.append(loss)
        delta = (
            0
            if loss == 0
            else tolerance
            if prev_loss is None
            else (prev_loss - loss) / loss
        )
        if delta < 0:
            raise ValueError("Loss increases. Decrease the learning rate")
        if delta < tolerance:
            theta = current_theta
            break
        prev_loss = loss
    return (progress, theta)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _normalize(x):
    """
    Obsolete. Use sklearn.preprocessing
    """
    means = [x[c].mean() for c in x.columns]
    # TODO: consider width = 0
    widths = [np.ptp(x[c]) for c in x.columns]
    result = (x - means) / widths
    return result, means, widths
