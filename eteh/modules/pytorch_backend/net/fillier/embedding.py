import torch
from .convblock import ConvBlock, ConvBlockFinal


class EmbeddingModel(torch.nn.Module):
    def __init__(self, input_channel, delta_channel, final_channel, dropout_rate):
        super(EmbeddingModel, self).__init__()
        self.conv = torch.nn.Sequential(
            ConvBlock(input_channel, delta_channel, dropout_rate),
            ConvBlock(delta_channel, 2*delta_channel, dropout_rate),
            ConvBlock(2*delta_channel, 3*delta_channel, dropout_rate),
            ConvBlock(3*delta_channel, 4*delta_channel, dropout_rate),
            ConvBlock(4*delta_channel, final_channel, dropout_rate),
            ConvBlockFinal(96, 96, dropout_rate)
        )
    
    def forward(self, x):
        return self.conv(x)

