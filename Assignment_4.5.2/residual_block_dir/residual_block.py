import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.gelu = torch.nn.GELU()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                            kernel_size=3, stride=stride, padding=1),
            torch.nn.BatchNorm2d(num_features=output_channels),
            self.gelu,
            torch.nn.Conv2d(in_channels=output_channels, out_channels=output_channels,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=output_channels),
        )

    def forward(self, x):
        skip_conn = x
        if self.downsample:
            skip_conn = self.downsample(x)
        x = self.layers(x)
        x += skip_conn
        return self.gelu(x)
