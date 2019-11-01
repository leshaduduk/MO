import pandas as pd
import numpy as np


def create_hypothesis(theta, poly=None, scaler=None):
    def hypothesis(sample):
        fitted = fit(sample, poly, scaler)
        return linear_hypothesis(theta, fitted)

    return hypothesis


def fit(sample, poly=None, scaler=None, scaler_fit=True):
    if poly is not None:
        sample = poly.transform(sample)
    else:
        sample = np.insert(sample, 0, 1, axis=1)
    if scaler is not None:
        sample = pd.DataFrame(sample)
        if scaler_fit:
            sample[sample.columns[1:]] = scaler.transform(sample[sample.columns[1:]])
        else:
            sample[sample.columns[1:]] = scaler.fit_transform(
                sample[sample.columns[1:]]
            )
    return sample


def gradient_descent(theta, x, y, alpha, reg_param):
    gradient = lambda _theta: linear_loss_gradient(_theta, x, y, reg_param)
    for current_theta in iterate_theta(gradient, theta, alpha):
        loss = linear_loss(current_theta, x, y, reg_param)
        yield current_theta, loss


def iterate_theta(gradient, theta, alpha):
    while True:
        yield theta
        theta = theta - alpha * gradient(theta)


def linear_hypothesis(theta, x):
    return np.dot(x, theta)


def linear_loss(theta, x, y, reg_param):
    prediction = linear_hypothesis(theta, x)
    cost_sum = quadratic_cost(prediction, y)
    reg_sum = reg_param * np.power(theta[1:], 2).sum()
    return 0.5 / len(x) * (cost_sum + reg_sum)


def linear_loss_gradient(theta, x, y, reg_param):
    cost_sum = np.dot(x.transpose(), linear_hypothesis(theta, x) - y)
    reg_sum = np.multiply(reg_param, theta)
    reg_sum[0] = 0
    return 1 / len(x) * (cost_sum + reg_sum)


def quadratic_cost(prediction, actual):
    return np.power(np.subtract(prediction, actual), 2).sum()


def run_descent(alpha, tolerance, theta_0, x, y, reg_param):
    prev_loss = None
    theta = None
    progress = []
    for current_theta, loss in gradient_descent(theta_0, x, y, alpha, reg_param):
        progress.append(loss)
        delta = tolerance if prev_loss is None else (prev_loss - loss)
        if delta < 0:
            raise ValueError("Loss increases. Decrease the learning rate")
        if delta < tolerance:
            theta = current_theta
            break
        prev_loss = loss
    return (progress, theta)
