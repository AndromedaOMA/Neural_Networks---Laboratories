import torch
import torchvision.datasets as datasets  # for Mist
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # To inherit our neural network
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader  # For management of the dataset (batches)
from tqdm import tqdm  # For nice progress bar!
import pandas as pd

"""
Best acc: 99.40%
"""


batch_size = 50
# total_epochs = 100
total_epochs = 200
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
            nn.LeakyReLU(negative_slope=0.001),
            nn.Dropout(0.1),
            # nn.ReLU(),
            nn.Linear(261, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(negative_slope=0.001),
            nn.Dropout(0.1),
            # nn.ReLU(),
            nn.Linear(200, classes_size),
            # nn.Dropout(0.1),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.layers(x)


# Load Data
train_transform = transforms.Compose([
    # transforms.RandomRotation(10),
    transforms.RandomAffine(2, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./training", train=True, transform=train_transform, download=True)
test_dataset = datasets.MNIST(root="./tests", train=False, transform=test_transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")

# Initialize network
model = NN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs
# optimizer = optim.SGD([torch.rand((2, 2), requires_grad=True)], lr=lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=total_epochs,
    steps_per_epoch=len(train_loader)
)


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


# Early stopping variables
patience = 15
best_test_acc = 0.0
epochs_without_improvement = 0
checkpoint_path = 'best_model.pth'


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

        scheduler.step()

    # scheduler.step(loss)

    # Evaluate accuracy on train and test set
    train_acc = check_accuracy(train_loader, model)
    test_acc = check_accuracy(test_loader, model)

    current_lr = scheduler.get_last_lr()[0]
    print(f"Train accuracy: {check_accuracy(train_loader, model)}%,"
          f"Test accuracy: {check_accuracy(test_loader, model)}%, "
          f"Current learning rate: {current_lr}")

    # Check for improvement in test accuracy
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        epochs_without_improvement = 0
        # Save model checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        print(f"New best test accuracy: {best_test_acc:.2f}%. Model saved.")
    else:
        epochs_without_improvement += 1

    # Early stopping condition
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered. No improvement in test accuracy for {patience} epochs.")
        break

# After training, load the model with the best test accuracy
model.load_state_dict(torch.load(checkpoint_path))
print("Model loaded from checkpoint with the best test accuracy.")

# Predict on test dataset and save to CSV
csv_data = {
    "ID": [],
    "target": [],
}

model.eval()
with torch.no_grad():
    for i, (data, targets) in enumerate(test_loader):
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
