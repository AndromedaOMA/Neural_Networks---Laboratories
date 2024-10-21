import numpy as np
from torchvision.datasets import MNIST
import time


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


def compute_softmax_function(w_vector):
    exp_element = np.exp(w_vector - np.max(w_vector, axis=1, keepdims=True))
    return exp_element / np.sum(exp_element, axis=1, keepdims=True)


def compute_cross_entropy(labels, probabilities):
    return -np.mean(np.sum(labels * np.log(probabilities + 1e-9), axis=1))


weight_matrix, bias = weight_matrix_and_bias_generator()
start_time = time.time()
for epoch in range(300):

    total_loss = 0
    for data_batch, label_batch in batches_generator(train_x, train_y):
        weight_vector = compute_weighted_sum(data_batch, weight_matrix, bias)
        softmax_probabilities = compute_softmax_function(weight_vector)

        predicted_class_index = np.argmax(softmax_probabilities, axis=1)
        target_class = np.argmax(label_batch, axis=1)

        loss = compute_cross_entropy(label_batch, softmax_probabilities)
        total_loss += loss

        gradients = softmax_probabilities - label_batch
        weight_matrix -= 0.01 * np.dot(data_batch.T, gradients)
        bias -= 0.01 * np.mean(gradients, axis=0, keepdims=True)

    # print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")
    # print(f"Epoch {epoch + 1}: bias = {bias}")

    if (epoch + 1) % 10 == 0:
        train_accuracy = np.mean(predicted_class_index == target_class)
        print(f"Epoch: {epoch + 1} -> Loss: {total_loss:.4f},  Train Accuracy: {train_accuracy:.4f}")


end_time = time.time()
print(f"Training Time: {end_time - start_time:.2f} seconds")


"""
inspiration:
    https://towardsdatascience.com/mnist-handwritten-digits-classification-from-scratch-using-python-numpy-b08e401c4dab
    https://www.geeksforgeeks.org/numpy-gradient-descent-optimizer-of-neural-networks/
"""