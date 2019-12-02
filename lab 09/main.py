import numpy as np
import pandas as pd


def cost_matrix(theta, x, y, r):
    return (x @ theta.T - y) * r


def cost_gradient(theta, x, y, r, reg_param):
    # NOTE: we should refactor to have a single gradient function
    cm = cost_matrix(theta, x, y, r)
    x_g = cm @ theta + reg_param * x
    theta_g = cm.T @ x + reg_param * theta
    return x_g, theta_g


def gradient_descent(theta, x, y, r, alpha, reg_param):
    gradient = lambda _x, _theta: cost_gradient(_theta, _x, y, r, reg_param)
    for x, theta in iterate(gradient, theta, x, alpha):
        cost = quadratic_cost(theta, x, y, r, reg_param)
        yield x, theta, cost


def iterate(gradient, theta, x, alpha):
    while True:
        yield x, theta
        x_g, theta_g = gradient(x, theta)
        x = x - alpha * x_g
        theta = theta - alpha * theta_g


def quadratic_cost(theta, x, y, r, reg_param):
    cm = cost_matrix(theta, x, y, r) ** 2
    reg_sum = (theta ** 2).sum().sum() + (x ** 2).sum().sum()
    # why does Andrew Ng tell to drop the 1/len?
    return 0.5 * (cm.sum().sum() + reg_param * reg_sum)


def run_descent(alpha, tolerance, theta_0, x_0, y, r, reg_param):
    prev_cost = None
    for x, theta, cost in gradient_descent(theta_0, x_0, y, r, alpha, reg_param):
        yield x, theta, cost
        delta = tolerance if prev_cost is None else (prev_cost - cost)
        if delta < 0:
            raise ValueError("Loss increases. Decrease the learning rate")
        if delta < tolerance:
            break
        prev_cost = cost
