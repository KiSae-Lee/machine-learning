from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load the MNIST dataset using torchvision.datasets
BATCH_SIZE = 32
train_set = datasets.MNIST(
    root="./mnist", train=True, transform=ToTensor(), download=True
)

test_set = datasets.MNIST(
    root="./mnist", train=False, transform=ToTensor(), download=True
)

print(len(train_set))

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

for x_train, t_train in train_loader:
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(x_train[i].reshape(28, 28), cmap="gray", interpolation="none")
        plt.title("digit: {}".format(t_train[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show()
    break


# Better! load the MNIST dataset using tensorflow
import keras
import matplotlib.pyplot as plt

Mnist = keras.datasets.mnist
(x_train, t_train), (x_test, t_test) = Mnist.load_data()

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.tight_layout()
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray", interpolation="none")
    plt.title("digit: {}".format(t_train[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()
