import numpy as np
import matplotlib.pyplot as plt


def function_2(x):
    # return x[0] ** 2 + x[1] ** 2
    return np.sum(x**2)


# Consider add type hints for numpy arrays
# reference: https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    init_x: initial value. 1D array.
    """
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def gradient_descent_recording(f, init_x, lr=0.01, step_num=100):
    """
    init_x: initial value. 1D array.
    """
    record = []
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        record.append(x.copy())

    return np.array(record)


init_x = np.array([-3.0, 4.0])
record = gradient_descent_recording(function_2, init_x, lr=0.1, step_num=100)

fig = plt.figure()
plt.scatter(record[:, 0], record[:, 1])
plt.xlim(-3, 3)
plt.ylim(-4, 4)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
