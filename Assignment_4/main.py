import torch
import torchvision.datasets as datasets # for Mist
import torchvision.transforms as transforms # Transformations we can perform on our dataset for augmentation
from torch import optim # For optimizers like SGD, Adam, etc.
from torch import nn # To inherit our neural network
from torch.utils.data import DataLoader # For management of the dataset (batches)
from tqdm import tqdm # For nice progress bar!


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def check_accuracy(loader, m):
    num_correct = 0
    num_samples = 0
    m.eval()

    with torch.no_grad():
        # Loop through the data
        for x, y in loader:
            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            # Get to correct shape
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 20

# Load Data
train_dataset = datasets.MNIST(
root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(
root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Train and validate the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device).reshape(data.shape[0], -1)
        targets = targets.to(device=device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Training loss and accuracy
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")
    train_acc = check_accuracy(train_loader, model)
    test_acc = check_accuracy(test_loader, model)
    print(f"Train Accuracy: {train_acc*100:.2f}%, Test Accuracy: {test_acc*100:.2f}%")


