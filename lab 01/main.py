import pandas as pd
import numpy as np


def analytic_solution(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


def create_design_matrix(*xs):
    ones = pd.Series(1, index=np.arange(len(xs[0])))
    result = pd.concat([ones, *xs], axis="columns")
    result.columns = np.arange(len(result.columns))
    return result


def gradient_descent(theta, x, y, alpha):
    gradient = lambda theta_: linear_loss_gradient(theta_, x, y)
    for current_theta in iterate_theta(gradient, theta, alpha):
        loss = linear_loss_function(current_theta, x, y)
        yield current_theta, loss


def iterate_theta(gradient, theta, alpha):
    while True:
        yield theta
        theta = theta - alpha * gradient(theta)


def linear_hypothesis(theta, xi):
    return xi.dot(theta)


def linear_loss_function(theta, x, y):
    hypothesis = lambda xi: linear_hypothesis(theta, xi)
    return loss_function(hypothesis, x, y)


def linear_loss_gradient(theta, x, y):
    # NOTE: x and theta should be of the same size: assuming a row of 1's in x
    len_x = len(x)
    if len_x != len(y):
        raise ValueError("Series must be of the same length")
    sample_items = x.transpose() * (x.dot(theta) - y).values
    return 1 / len_x * sample_items.apply(np.sum, axis="columns")


def loss_function(hypothesis, x, y):
    len_x = len(x)
    if len_x != len(y):
        raise ValueError("Series must be of the same length")
    return 0.5 / len_x * sum((hypothesis(xi) - yi) ** 2 for xi, yi in zip(x.values, y))


def normalize(x):
    def _normalize(row):
        mean = row.mean()
        width = row.max() - row.min()
        return row.map(lambda item: (item - mean) / width)

    return x.apply(_normalize, axis="rows")
