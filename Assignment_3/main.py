import numpy as np
from torchvision.datasets import MNIST
import time
import matplotlib.pyplot
# %matplotlib inline


def download_mnist(is_train: bool):
    """DOWNLOAD AND SET OUT DATA SET OF HAND WRITTEN IMAGES + LABELS"""
    dataset = MNIST(root='./data', transform=lambda x: np.array(x).flatten(), download=True, train=is_train)

    mnist_data = []
    mnist_labels = []

    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    mnist_data = np.array(mnist_data)
    mnist_labels = np.array(mnist_labels)

    encoded_labels = np.zeros((len(mnist_labels), max(mnist_labels)+1), dtype=int)
    encoded_labels[np.arange(len(mnist_labels)), mnist_labels] = 1
    return mnist_data, encoded_labels


train_x, train_y = download_mnist(True) #60000
test_x, test_y = download_mnist(False)  #10000


print(f"Here we got the length of the 99th training data: {len(train_x[99])},\n"
      f" The 99th training data: {train_x[99]},\n The 99th training label: {train_y[99]},\n")


def batches_generator(train_data, train_labels, no_of_batches):
    """YIELD (continuous "return") the current batches of 100 elements each"""
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)

    for i in range(0, len(train_data), no_of_batches):
        batch_indices = indices[i:i + no_of_batches]
        yield train_data[batch_indices], train_labels[batch_indices]


class NN:
    def __init__(self, sizes=None, epochs=10, batches=100, learning_rate=0.1):
        if sizes is None:
            sizes = [784, 100, 10]
        self.sizes = sizes
        self.epochs = epochs
        self.batches = batches
        self.learning_rate = learning_rate

        in_layer = self.sizes[0]
        h_layer = self.sizes[1]
        out_layer = self.sizes[2]

        """configurare Xavier pentru model ce utilizează funcții simetrice
           (tocmai pentru evitarea blocării rețelei neuronale)"""
        self.params = {
            'W1': np.random.randn(h_layer, in_layer) * np.sqrt(2 / (h_layer + in_layer)),
            'W2': np.random.randn(out_layer, h_layer) * np.sqrt(2 / (out_layer + h_layer))
        }

    def forward_prop(self):
        pass

    def backward_prop(self):
        pass

    def compute_acc(self):
        pass

    def update_w(self):
        pass

