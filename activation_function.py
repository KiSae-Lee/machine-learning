import numpy as np
import matplotlib.pylab as plt


def step_function(x):  # Only takes a number
    if x > 0:
        return 1
    else:
        return 0


def step_function_array(x):  # It can takes an array
    y = x > 0  # It will compare each element of the array with 0
    return y.astype(np.int_)  # It will convert the boolean array to int array


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


x1 = np.arange(-5.0, 5.0, 0.1)
y1 = step_function_array(x1)

x2 = np.arange(-5.0, 5.0, 0.1)
y2 = sigmoid(x2)

x3 = np.arange(-5.0, 5.0, 0.1)
y3 = relu(x3)

plt.plot(x1, y1, linestyle="--", label="step")
plt.plot(x2, y2, label="sigmoid")
plt.plot(x3, y3, label="ReLU")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
