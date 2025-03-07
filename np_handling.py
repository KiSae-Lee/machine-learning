import numpy as np

A = np.array([[1, 2, 3, 4]])
print(np.ndim(A))  # 1. Returns the number of dimensions of the array
print(A.shape)  # (4,). Returns the shape of the array

B = np.array([[1, 2], [3, 4], [5, 6]])
print(np.ndim(B))  # 2
print(B.shape)  # (3, 2)

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])
C = np.dot(A, B)
print(C)  # [[22 28] [49 64]]

# In neural network
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)
print(Y)  # [5 11 17]
