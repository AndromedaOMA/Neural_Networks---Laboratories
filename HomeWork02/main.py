import numpy as np
from torchvision.datasets import MNIST


def download_mnist(is_train: bool):
    """DOWNLOAD AND SET OUT DATA SET OF HAND WRITTEN IMAGES + LABELS"""
    dataset = MNIST(root='./data', transform=lambda x: np.array(x).flatten(), download=True, train=is_train)

    mnist_data = []
    mnist_labels = []

    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    encoded_labels = np.zeros((len(mnist_labels), max(mnist_labels)+1), dtype=int)
    encoded_labels[np.arange(len(mnist_labels)), mnist_labels] = 1
    return mnist_data, encoded_labels


train_x, train_y = download_mnist(True) #60000
test_x, test_y = download_mnist(False)  #10000


def batches_generator(train_data, train_labels):
    """YIELD (continuous "return") the current baatches of 100 elements each"""
    np.random.shuffle(train_data)
    np.random.shuffle(train_labels)

    for i in range(0, len(train_data), 100):
        yield train_data[i:i + 100], train_labels[i:i + 100]

for epoch in range(250):
    for data_batch, label_batch in batches_generator(train_x, train_y):
        #TO DO