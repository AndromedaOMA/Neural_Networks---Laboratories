import torch


class DuelingCNN(torch.nn.Module):
    def __init__(self, input_channels, input_size=64, hid_layer_dim=256, out_layer_dim=2):
        super(DuelingCNN, self).__init__()
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
            torch.nn.Linear(conv_output_size, hid_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        )
        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(hid_layer_dim, hid_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_layer_dim, 1)
        )
        self.advantages_stream = torch.nn.Sequential(
            torch.nn.Linear(hid_layer_dim, hid_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_layer_dim, out_layer_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        # Value
        V = self.value_stream(x)
        # Advantages
        A = self.advantages_stream(x)
        # Q value (Dueling DQN Formula: V + A - mean(A))
        Q = V + A - torch.mean(A, dim=1, keepdim=True)
        return Q

