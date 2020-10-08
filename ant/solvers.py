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


class Linear1DSolver(Solver):
    def __init__(self, in_shape, num_classes):
        super().__init__(in_shape, num_classes)
        # assert len(in_shape) == 1

        if num_classes == 1:
            self.model = nn.Linear(in_shape[0], 1)
        else:
            self.model = nn.Sequential(
                nn.Linear(np.prod(in_shape), num_classes), nn.Softmax(dim=1)
            )

    def forward(self, x):
        return self.model(x)


class Linear2DSolver(Solver):
    def __init__(self, in_shape, num_classes):
        super().__init__(in_shape, num_classes)
        # assert len(in_shape) == 2

        if num_classes == 1:
            self.model = nn.Linear(in_shape[0], 1)
        else:
            self.model = nn.Sequential(
                nn.Linear(np.prod(in_shape), num_classes), nn.Softmax(dim=1)
            )

    def forward(self, x):
        return self.model(torch.flatten(x, start_dim=1))
