import torch

class ConvBlock(torch.nn.Module):

    def __init__(self, channel_in, channel_out, dropout_rate):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(channel_in, channel_out, kernel_size=(1, 3), stride=1, padding=(0,1)),
            torch.nn.Conv2d(channel_out, channel_out, kernel_size=(3, 1), stride=1, padding=(1,0)),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(channel_out, channel_out, kernel_size=(3, 1), stride=1, padding=(1,0)),
            torch.nn.Conv2d(channel_out, channel_out, kernel_size=(1, 3), stride=1, padding=(0,1)),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU()
        )
    def forward(self, x):
        assert x.size(-1) >=2
        assert x.size(-2) >=2
        return self.conv(x)

class ConvBlockFinal(torch.nn.Module):

    def __init__(self, channel_in, channel_out, dropout_rate):
        super(ConvBlockFinal, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(channel_in, channel_out, kernel_size=(3, 1), stride=1, padding=(1,0)),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channel_out, channel_out, kernel_size=(3, 1), stride=1, padding=(1,0)),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    x = torch.Tensor(1, 1, 196, 32)
    print(x.size())
    conv = torch.nn.Sequential(
        ConvBlock(1,24,0.1),
        ConvBlock(24,48,0.1),
        ConvBlock(48,72,0.1),
        ConvBlock(72,96,0.1),
        ConvBlock(96,96,0.1),
        ConvBlockFinal(96,96,0.1)
    )
    y = conv(x)
    print(y.size())

    pass