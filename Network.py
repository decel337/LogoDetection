import torch

import torch.nn.functional as F
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, layerList):
        super(Network, self).__init__()
        self.functions = []
        layers = []
        for layer in layerList:
            self.functions.append(self.makeLayer(layer))

        for layer in self.functions:
            if not isinstance(layer, list):
                layers.append(layer)

        self.layer = nn.ModuleList(layers)

    def forward(self, x):
        layerNum = 0
        for layer in self.functions:
            if isinstance(layer, list):
                if layer[0] == "ReLu":
                    x = F.relu(x)
                elif layer[0] == "Sigmoid":
                    x = torch.sigmoid(x)
                elif layer[0] == "Max Pool":
                    x = F.max_pool2d(x, layer[1])
                elif layer[0] == "Avg Pool":
                    x = F.avg_pool2d(x, layer[1])

                elif layer[0] == "View":
                    x = x.view(-1, layer[1])
                else:
                    pass
            else:
                x = self.layer[layerNum](x)
                layerNum += 1
        return x

    def makeLayer(self, layerCfg):
        if layerCfg[0] == "Linear Layer":
            return nn.Linear(layerCfg[1], layerCfg[2])
        elif layerCfg[0] == "Convolution Layer":
            return nn.Conv2d(layerCfg[1], layerCfg[2], kernel_size=layerCfg[3])
        elif layerCfg[0] == "Dropout":
            return nn.Dropout2d()
        elif layerCfg[0] in ["View", "Avg Pool", "Max Pool"]:
            return [layerCfg[0], layerCfg[1]]
        else:
            return [layerCfg[0]]
