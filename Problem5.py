import math
import random
import numpy as np


def objective_function(data, y, w, delta):
    f = 0
    for j in range(0, 100):
        wx = (w[0] * data[j][0]) + (w[1] * data[j][1])
        if y[j] >= wx + delta:
            f += (y[j] - wx - delta) ** 2
        elif abs(y[j] - wx) < delta:
            f += 0
        elif y[j] <= wx - delta:
            f += (y[j] - wx + delta) ** 2
    return f / 100


def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    return new_w, history_fw


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
    return new_w, history_fw
