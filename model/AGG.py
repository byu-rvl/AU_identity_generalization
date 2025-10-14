import torch
from .basic_block import *


class AGG(nn.Module):
    def __init__(self, num_classes, in_channels, secondDimensionSize, numEncoderLayers, head_embedding_dim):
        super(AGG, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.numberHeads = self.num_classes + self.num_classes * self.num_classes

        self.change_dimension = PositionalBlock(in_channels, self.numberHeads * head_embedding_dim)

        self.head_embedder = nn.Embedding(self.numberHeads, head_embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=head_embedding_dim, nhead=int(self.num_classes))
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=numEncoderLayers)

    def getAGG(self):
        return self.change_dimension, self.head_embedder, self.transformer_encoder, self.numberHeads