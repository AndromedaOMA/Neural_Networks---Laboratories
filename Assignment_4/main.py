import torch
import torchvision.datasets as datasets  # for Mist
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # To inherit our neural network
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader  # For management of the dataset (batches)
from tqdm import tqdm  # For nice progress bar!
import pandas as pd


batch_size = 50
total_epochs = 50
lr = 0.001


class NN(nn.Module):
    def __init__(self, input_size=784, classes_size=10):
        super(NN, self).__init__()
        self.layers = torch.nn.Sequential(
            nn.Linear(input_size, 522),
            nn.BatchNorm1d(522),
            nn.ReLU(),
            nn.Linear(522, 261),
            nn.BatchNorm1d(261),
            nn.ReLU(),
            nn.Linear(261, 174),
            nn.BatchNorm1d(174),
            nn.ReLU(),
            nn.Linear(174, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 87),
            nn.BatchNorm1d(87),
            nn.ReLU(),
            nn.Linear(87, classes_size),
            # nn.Dropout(0.1),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.layers(x)


# Load Data
train_transform = transforms.Compose([
    # transforms.RandomRotation(2),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

train_dataset = datasets.MNIST(root="./training", train=True, transform=train_transform, download=True)
test_dataset = datasets.MNIST(root="./tests", train=False, transform=test_transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Set device cuda for GPU if it's available otherwise run on the CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

# Initialize network
model = NN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs
# optimizer = optim.SGD([torch.rand((2, 2), requires_grad=True)], lr=lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)


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

    scheduler.step(loss)

    current_lr = scheduler.get_last_lr()[0]
    print(f"Train accuracy: {check_accuracy(train_loader, model)}%,"
          f"Test accuracy: {check_accuracy(test_loader, model)}%, "
          f"Current learning rate: {current_lr}")

# Predict on test dataset and save to CSV
csv_data = {
    "ID": [],
    "target": [],
}

model.eval()
with torch.no_grad():
    for i, (data, targets) in enumerate(tqdm(test_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape
        data = data.reshape(data.shape[0], -1)

        # Forward pass
        scores = model(data)
        _, predictions = scores.max(1)

        # Log ID and predicted targets
        for j in range(len(data)):
            csv_data["ID"].append((i * batch_size) + j)
            csv_data["target"].append(predictions[j].item())

# Save to CSV
df = pd.DataFrame(csv_data)
df.to_csv("./submission.csv", index=False)
