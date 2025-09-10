import torch
from .basic_block import *


class AGG(nn.Module):
    def __init__(self, num_classes, in_channels, secondDimensionSize, numEncoderLayers):
        super(AGG, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.decrease_dim = PositionalBlock(in_channels*secondDimensionSize, self.in_channels*2)

        positional_encoding = []
        numberHeads = self.num_classes + self.num_classes * self.num_classes
        for i in range(numberHeads):
            positional_encode_layer = PositionalBlock(self.in_channels*2, self.in_channels)
            positional_encoding += [positional_encode_layer]

        self.positional_encoding = nn.ModuleList(positional_encoding)

        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=int(self.num_classes))
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=numEncoderLayers)

    def getAGG(self):
        return self.decrease_dim, self.positional_encoding, self.transformer_encoder