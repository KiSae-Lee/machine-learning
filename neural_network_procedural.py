import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# this function is not doing anything
# in this specific case
def identity_function(x):
    return x


# input layer to level 1
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(Z1)  # [0.57444252 0.66818777 0.75026011]

# level 1 to level 2
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)  # [0.62624937 0.7710107]

# level 2 to output layer
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print(Y)  # [0.31682708 0.69627909]
