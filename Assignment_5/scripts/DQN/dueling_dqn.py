import torch


class DuelingDQN(torch.nn.Module):
    def __init__(self, in_layer_dim=12, hid_layer_dim=256, out_layer_dim=2):
        super(DuelingDQN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_layer_dim, hid_layer_dim),
            torch.nn.LayerNorm(hid_layer_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),
            # torch.nn.Linear(hid_layer_dim, out_layer_dim)
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
        x = self.layers(x)
        # Value
        V = self.value_stream(x)
        # Advantages
        A = self.advantages_stream(x)
        # Q value (Dueling DQN Formula)
        Q = V + A - torch.mean(A, dim=1, keepdim=True)
        return Q
