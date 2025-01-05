import torch


class DQN(torch.nn.Module):
    def __init__(self, in_layer_dim=12, hid_layer_dim=256, out_layer_dim=2):
        super(DQN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_layer_dim, hid_layer_dim),
            torch.nn.LayerNorm(hid_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_layer_dim, hid_layer_dim),
            torch.nn.LayerNorm(hid_layer_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(hid_layer_dim, out_layer_dim)
        )
        """
        For trained_q_function_14.000:
            torch.nn.Linear(in_layer_dim, hid_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_layer_dim, hid_layer_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hid_layer_dim, out_layer_dim),
        """

    def forward(self, x):
        x = self.layers(x)
        return x
