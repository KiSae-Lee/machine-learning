import keras
import numpy as np


def get_data():
    Mnist = keras.datasets.mnist
    (x_train, t_train), (x_test, t_test) = Mnist.load_data()
    # Normalize the image data
    return x_train.reshape([-1, 28 * 28]) / 255, t_train


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


x_train, t_train = get_data()

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)  # indices of the mini-batch

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch.shape)
print(t_batch.shape)

print(cross_entropy_error(x_batch, t_batch))
