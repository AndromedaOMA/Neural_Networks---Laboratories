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

    mnist_data = (np.array(mnist_data) / 255.0 * 0.99) + 0.01
    mnist_labels = np.array(mnist_labels)

    encoded_labels = np.zeros((len(mnist_labels), max(mnist_labels) + 1), dtype=int)
    encoded_labels[np.arange(len(mnist_labels)), mnist_labels] = 1
    return mnist_data, encoded_labels


train_x, train_y = download_mnist(True)  #60000
test_x, test_y = download_mnist(False)  #10000
print(f"Here we got the #row  {len(train_x)} and #col: {len(train_x[0])} of training data.\n")
print(f"Here we got the length of the 99th training data: {len(train_x[99])},\n"
      f" The 99th training data: {train_x[99]},\n The 99th training label: {train_y[99]},\n")


def batches_generator(train_data, train_labels, no_of_batches):
    """YIELD (continuous "return") the current batches of 100 elements each"""
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)

    for i in range(0, len(train_data), no_of_batches):
        batch_indices = indices[i:i + no_of_batches]
        yield train_data[batch_indices], train_labels[batch_indices]


def sigmoid(x, backpropagation=False):
    # Clip values to prevent overflow
    x = np.clip(x, -500, 500)  # Clip x to avoid large values
    s = 1 / (1 + np.exp(-x))
    if backpropagation:
        return s * (1 - s)
    return s


def softmax(x, backpropagation=False):
    exp = np.exp(x - np.max(x))
    s = exp / np.sum(exp, axis=0)
    if backpropagation:
        return s * (1 - s)
    return s


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
            'W1': np.random.randn(h_layer, in_layer) * np.sqrt(2 / (h_layer + in_layer)),  #100x784
            'W2': np.random.randn(out_layer, h_layer) * np.sqrt(2 / (out_layer + h_layer))  #10x100
        }

    def forward_prop(self, x_train):
        params = self.params

        params['A0'] = x_train  #784x1
        params['Z1'] = np.dot(params['W1'], params['A0'])  #100x1
        params['A1'] = sigmoid(params['Z1'])  #100x1
        params['Z2'] = np.dot(params['W2'], params['A1'])  #10x1
        params['A2'] = softmax(params['Z2'])  #10x1

        return params['A2']

    def backward_prop(self, y_train, output):
        params = self.params

        # err = output - y_train
        err = (output - y_train) * softmax(params['Z2'], backpropagation=True)
        params['W2'] -= self.learning_rate * np.outer(err, params['A1'])

        err = np.dot(params['W2'].T, err) * sigmoid(params['Z1'], backpropagation=True)
        params['W1'] -= self.learning_rate * np.outer(err, params['A0'])

    def compute_acc(self, test_data, test_labels):
        predictions = []
        # for data_batch, label_batch in batches_generator(test_data, test_labels, self.batches):
        for i in range(len(test_data)):
            output = self.forward_prop(test_data[i])
            predict = np.argmax(output)
            predictions.append(predict == np.argmax(test_labels[i]))
        return np.mean(predictions)

    def train(self, train_list, train_labels, test_list, test_labels):
        start_time = time.time()
        for i in range(self.epochs):
            # for data_batch, label_batch in batches_generator(train_list, train_labels, self.batches):
            for j in range(len(train_list)):
                output = self.forward_prop(train_list[j])
                self.backward_prop(train_labels[j], output)

            accuracy = self.compute_acc(test_list, test_labels)
            print(f'Epoch: {i + 1}, Time Spent: {time.time() - start_time}s, Accuracy: {accuracy * 100}%')


if __name__ == "__main__":
    nn = NN()
    nn.train(train_x, train_y, test_x, test_y)

