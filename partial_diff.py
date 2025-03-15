import numpy as np
import matplotlib.pyplot as plt


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # generate array with same shape as x

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # restore value

    return grad


# input is 1D array with 2 elements
def function_2(x):
    # return x[0] ** 2 + x[1] ** 2
    return np.sum(x**2)


x = np.linspace(-3, 3, 25)
y = np.linspace(-3, 3, 25)
X, Y = np.meshgrid(x, y)

# TODO: Need to figure out what is happening here
Z = np.array(
    [
        [function_2(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)]
        for row_x, row_y in zip(X, Y)
    ]
)

# TODO: Need to figure out what is happening here
grad = np.array(
    [
        [
            numerical_gradient(function_2, np.array([xi, yi]))
            for xi, yi in zip(row_x, row_y)
        ]
        for row_x, row_y in zip(X, Y)
    ]
)

# Extract gradient components
U = grad[:, :, 0]  # Gradient in x direction
V = grad[:, :, 1]  # Gradient in y direction

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")

plt.show()

fig = plt.figure()
# Quiver graph
plt.quiver(X, Y, U, V)

plt.show()
