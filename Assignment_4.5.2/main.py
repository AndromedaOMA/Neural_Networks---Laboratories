import torch.nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import pandas as pd
from residual_block_dir import ResidualBlock

"""
Best acc: ?? %
"""

total_epochs = 200
batch_size = 100


class ResNet(torch.nn.Module):
    def __init__(self, input_channels=1, input_size=28, output_size=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.residual_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            self._make_layer(ResidualBlock, 16, 2),
            self._make_layer(ResidualBlock, 32, 2, stride=2))

        residual_output_size = 32 * (input_size // 4) ** 2
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(residual_output_size, 150),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(150, output_size),
            torch.nn.LogSoftmax(dim=1))

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.residual_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x


# device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transformer = transforms.Compose([
    transforms.RandomAffine(degrees=2, translate=[0.1, 0.1]),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])
test_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

train_dataset = datasets.MNIST(root='./train', train=True, transform=train_transformer, download=True)
test_dataset = datasets.MNIST(root='./test', train=False, transform=test_transformer, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    model.train()
    return (num_correct / num_samples) * 100


model = ResNet().to(device)

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


patience = 20
best_test_acc = 0
epochs_without_improvements = 0
checkpoint_path = './best_model.pth'

for epoch in range(total_epochs):
    print(f"Epoch {epoch+1}/{total_epochs}")
    for _, (data, labels) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        labels = labels.to(device)

        scores = model(data)
        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

    train_acc = check_accuracy(train_loader, model)
    test_acc = check_accuracy(test_loader, model)

    current_lr = scheduler.get_last_lr()[0]
    print(f"Train accuracy: {train_acc}%,"
          f"Test accuracy: {test_acc}%, "
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
