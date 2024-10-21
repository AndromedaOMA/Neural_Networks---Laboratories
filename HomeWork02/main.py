#inspiration: https://towardsdatascience.com/mnist-handwritten-digits-classification-from-scratch-using-python-numpy-b08e401c4dab


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
    """YIELD (continuous "return") the current batches of 100 elements each"""
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)

    for i in range(0, len(train_data), 100):
        batch_indices = indices[i:i + 100]
        yield train_data[batch_indices], train_labels[batch_indices]


def weight_matrix_and_bias_generator():
    return np.random.random((784, 10)) * 0.01,  np.random.random((1, 10))


def compute_weighted_sum(data, weight, bias_vector):
    return np.dot(data, weight) + bias_vector


def compute_softmax_function(weight_vector):
    exp_element = np.exp(weight_vector - np.max(weight_vector, axis=1))
    return exp_element / np.sum(exp_element, axis=1)


def compute_cross_entropy(labels, probabilities):
    return -np.sum(labels * np.log(probabilities), axis=1)


def gradient_descent()


weight_matrix, bias = weight_matrix_and_bias_generator()
for epoch in range(300):
    for data_batch, label_batch in batches_generator(train_x, train_y):
        z = compute_weighted_sum(data_batch, weight_matrix, bias)
        softmax_probabilities = compute_softmax_function(z)
        predicted_class_index = np.argmax(softmax_probabilities)
        loss = compute_cross_entropy(label_batch, softmax_probabilities)

