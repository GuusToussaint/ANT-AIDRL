import torch
import numpy as np
import torch.nn as nn


class Solver(nn.Module):
    """Given its input outputs its prediction for each class. When num_classes = 1,
    assume regression, otherwise output probabilities for each class."""

    def __init__(self, in_shape, num_classes):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = (1,)
        self.num_classes = num_classes


class LinearClassifier(Solver):
    def __init__(self, in_shape, num_classes, GAP=False):
        super().__init__(in_shape, num_classes)
        assert self.num_classes > 1

        nodes = np.prod(in_shape) if not GAP else in_shape[0]

        modules = []
        if GAP:
            modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(nn.Flatten())
        modules.append(nn.Linear(nodes, num_classes))
        modules.append(nn.LogSoftmax(dim=1))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class LinearRegressor(Solver):
    def __init__(self, in_shape, num_classes):
        super().__init__(in_shape, num_classes)
        nodes = np.prod(in_shape)

        modules = [nn.Flatten(), nn.Linear(nodes, num_classes)]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)
