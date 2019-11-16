import numpy as np
import pandas as pd


def choose_threshold(x, mean, variance, y, epsilon):
    scores = [f1_score(predict(x, mean, variance, e), y) for e in epsilon]
    if not has_maximum(scores):
        raise ValueError("The provided range does not contain the optimal value.")
    index, _ = max(enumerate(scores), key=lambda t: t[1])
    return epsilon[index]


def f1_score(prediction, actual):
    p = precision(prediction, actual)
    r = recall(prediction, actual)
    return 2 * p * r / (p + r)


def gaussian(x, mean, variance):
    factor = 1 / np.sqrt(2 * np.pi * variance)
    return factor * np.exp(-(x - mean) ** 2 / 2 / variance)


def gaussian_params(x):
    mean = 1 / (len(x) - 1) * x.sum(axis=0)
    variance = 1 / len(x) * ((x - mean) ** 2).sum(axis=0)
    return np.array(mean), np.array(variance)


def has_maximum(seq):
    return seq[0] < max(seq) > seq[-1]


def precision(prediction, actual):
    if len(prediction) != len(actual):
        raise ValueError("Inputs must be of the same size")
    positive = actual[actual == 1]
    positive_prediction = prediction[positive.index]
    return positive_prediction.sum() / prediction.sum()


def predict(x, mean, variance, epsilon):
    p = probability(x, mean, variance)
    return (p < epsilon).astype(int)


def probability(x, mean, variance):
    return np.product(gaussian(x, mean, variance), axis=1)


def recall(prediction, actual):
    if len(prediction) != len(actual):
        raise ValueError("Inputs must be of the same size")
    positive = actual[actual == 1]
    positive_prediction = prediction[positive.index]
    return positive_prediction.sum() / actual.sum()
