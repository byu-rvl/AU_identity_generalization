import torch
from torch import nn

class COAL(nn.Module):
    def __init__(self, num_classes, in_channels, numLandmarks):
        super(COAL, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.numLandmarks = numLandmarks
        
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.edge_fc = nn.Linear(self.in_channels, 4)
        self.relu = nn.ReLU()

        self.emb_layer = nn.Linear(in_channels*num_classes, in_channels)

        self.lmk_layer1 = nn.Linear(in_channels*num_classes, in_channels*num_classes*2)
        self.lmk_layer2 = nn.Linear(in_channels*num_classes*2, self.numLandmarks*2)

    def getInfo(self):
        return self.sc, self.edge_fc, self.relu, self.emb_layer, self.lmk_layer1, self.lmk_layer2