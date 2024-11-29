import torch
import torchvision.datasets as datasets  # for Mist
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # To inherit our neural network
from torch.utils.data import DataLoader  # For management of the dataset (batches)
from tqdm import tqdm  # For nice progress bar!


class NN(nn.Module):
    def __init__(self, input_size=784, classes_size=10):
        super(NN, self).__init__()
        self.layers = torch.nn.Sequential(nn.Linear(input_size, 100),
                                          nn.ReLU(),
                                          nn.Linear(100, classes_size),
                                          nn.Dropout(0.2))

    def forward(self, x):
        return self.layers(x)


# Load Data
image_to_tensor = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./training", train=True, transform=image_to_tensor, download=True)
test_dataset = datasets.MNIST(
    root="./tests", train=False, transform=image_to_tensor, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize network
model = NN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

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
            num_correct += (predictions == y).sum().item()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return (num_correct / num_samples) * 100


# Train and validate the model
total_epochs = 25
for epoch in range(total_epochs):
    print(f"Epoch: {epoch+1}/{total_epochs}")
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape
        data = data.reshape(data.shape[0], -1)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()

    print(f"Train accuracy: {check_accuracy(train_loader, model)}%, Test accuracy: {check_accuracy(test_loader, model)}%")

