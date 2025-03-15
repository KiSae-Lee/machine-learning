import numpy as np


def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


y = [
    0.1,
    0.05,
    0.6,
    0.0,
    0.05,
    0.1,
    0.0,
    0.1,
    0.0,
    0.0,
]  # output of the neural network
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # answer label

print(sum_squares_error(np.array(y), np.array(t)))  # 0.09750000000000003
print(cross_entropy_error(np.array(y), np.array(t)))  # 0.510825457099338
