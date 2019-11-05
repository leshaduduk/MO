from itertools import islice


import pandas as pd
import numpy as np


def activate(theta, a):
    bias, weights = np.hsplit(theta, [1])
    return sigmoid(np.dot(weights, a) + bias.reshape(-1))


def network_cost(thetas, a, y, reg_param):
    cost_sum = log_loss(a, y)
    reg_sum = reg_param / 2 * sum_reg(thetas)
    return 1 / len(y) * (cost_sum + reg_sum)


def forward_propagate(thetas, x):
    activations = [np.array(x)]
    for theta in thetas:
        activation = activate(theta, activations[-1])
        activations.append(activation)
    return activations


def backpropagate(thetas, x, y, reg_param):
    deltas = [np.zeros(theta.shape) for theta in thetas]
    for xrow, yrow in zip(x.itertuples(index=False), y.itertuples(index=False)):
        activations = forward_propagate(thetas, xrow)
        error = activations[-1] - yrow
        for layer in range(len(activations) - 2, -1, -1):
            deltas[layer][:, 0] += error
            a_j, error_i = np.meshgrid(activations[layer], error)
            deltas[layer][:, 1:] += a_j * error_i
            # we compute error on the last iteration in vain
            error = np.dot(thetas[layer].T, error)[1:] * sigmoid_da(activations[layer])
    for d, theta in zip(deltas, thetas):
        _, weight = np.hsplit(theta, [1])
        d[:, 1:] += reg_param * weight
        yield 1 / len(x) * d


def iterate_thetas(deltas_func, thetas, alpha):
    while True:
        yield thetas
        deltas = deltas_func(thetas)
        thetas = [theta - alpha * d for d, theta in zip(deltas, thetas)]


def gradient_descent(thetas, x, y, alpha, reg_param):
    deltas_func = lambda _thetas: backpropagate(_thetas, x, y, reg_param)
    for current_thetas in iterate_thetas(deltas_func, thetas, alpha):
        # TODO: optimize? we calculate the same activation on the next iteration
        a = run_network(current_thetas, x)
        current_cost = network_cost(current_thetas, a, y, reg_param)
        yield current_thetas, current_cost


def gradient_approx(thetas, x, y, reg_param, epsilon, layers, rows, columns):
    result = [np.zeros(theta.shape) for theta in thetas]
    for layer in layers:
        theta = thetas[layer]
        for i in rows:
            for j in columns:
                original = theta[i, j]

                theta[i, j] = original + epsilon
                a = run_network(thetas, x)
                top = network_cost(thetas, a, y, reg_param)

                theta[i, j] = original - epsilon
                a = run_network(thetas, x)
                bottom = network_cost(thetas, a, y, reg_param)

                theta[i, j] = original
                result[layer][i, j] = (top - bottom) / 2 / epsilon
    return result


def run_descent(alpha, tolerance, thetas_0, x, y, reg_param):
    prev_loss = None
    thetas = None
    progress = []
    for current_thetas, loss in gradient_descent(thetas_0, x, y, alpha, reg_param):
        print("Cost:", loss)
        progress.append(loss)
        delta = tolerance if prev_loss is None else (prev_loss - loss)
        if delta < 0:
            # TODO?: the cost might increase sometimes. don't throw?
            raise ValueError("Cost increases. Decrease the learning rate")
        if delta < tolerance:
            thetas = current_thetas
            return (progress, thetas)
        prev_loss = loss


def run_stochastic_descent(alpha, epochs, thetas_0, x, y, reg_param, batch_size):
    indexes = list(range(len(x)))
    prev_loss = None
    thetas = thetas_0
    progress = []
    for i in range(epochs):
        np.random.shuffle(indexes)
        for start in range(len(indexes) // batch_size):
            batch_indexes = indexes[
                start * batch_size : start * batch_size + batch_size
            ]
            x_batch = x.iloc[batch_indexes]
            x_batch.reset_index(drop=True, inplace=True)
            y_batch = y.iloc[batch_indexes]
            y_batch.reset_index(drop=True, inplace=True)
            # the first value is the one we provided
            v = islice(gradient_descent(thetas, x_batch, y_batch, alpha, reg_param), 2)
            cost = None
            for thetas, cost in v:
                pass
            yield cost, thetas
        print("Epoch {} completed".format(i + 1))


def init_weights(levels, epsilon):
    thetas = []
    for i in range(1, len(levels)):
        # +1 accounts for bias units
        theta = np.random.random((levels[i], levels[i - 1] + 1))
        theta = 2 * epsilon * theta - epsilon
        thetas.append(theta)
    return thetas


def log_loss(prediction, y):
    result = y * np.log(prediction) + (1 - y) * np.log(1 - prediction)
    # pandas => need a sum for each dim
    return -result.sum().sum()


def run_network(thetas, x, layer=-1):
    # NOTE: assuming x is a dataframe
    result = []
    for row in x.itertuples(index=False):
        result.append(forward_propagate(thetas, row)[layer])
    return pd.DataFrame(result)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_da(a):
    return a * (1 - a)


def sum_reg(thetas):
    # NOTE: assuming theta is a dataframe
    # iloc excludes bias units
    return sum(
        np.power(pd.DataFrame(theta).iloc[:, 1:], 2).sum().sum() for theta in thetas
    )


def to_activation(cls, total):
    result = np.zeros(total)
    result[cls] = 1
    return result


def to_cls(a):
    return max(enumerate(a), key=lambda p: p[1])[0]
