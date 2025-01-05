import torch


class CNN(torch.nn.Module):
    def __init__(self, input_channels, input_size=64, out_layer_dim=2):
        super(CNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout2d(0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout2d(0.1)
        )
        conv_output_size = (input_size // 4) ** 2 * 32
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(conv_output_size, 150),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(150, out_layer_dim),
            torch.nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x
