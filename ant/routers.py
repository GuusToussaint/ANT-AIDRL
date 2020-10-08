import torch
import torch.nn as nn
import numpy as np
from math import floor


class Router(nn.Module):
    """Given it's input outputs a scalar representing a probability
    of using the left subtree in the ANT."""

    def __init__(self, in_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = in_shape


class FullyConnectedSigmoidRouter(Router):
    def __init__(self, in_shape):
        super().__init__(in_shape)
        # assert len(in_shape) == 1
        self.model = nn.Sequential(nn.Linear(np.prod(in_shape), 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(torch.flatten(x, start_dim=1))


class Conv2DFCSigmoid(Router):
    def __init__(
        self,
        in_shape,
        convolutions=1,
        fully_connected=1,
        kernels=40,
        kernel_size=5,
        pad=0,
        dilation=1,
        stride=1,
    ):
        # TODO: only works with square images
        in_shapes = [in_shape[0]]
        for convolution in range(convolutions):
            if convolution > 0:
                in_shapes.append(kernels)
        super().__init__(in_shape)

        self.model = nn.Sequential()
        for convolution in range(convolutions):
            conv = nn.Conv2d(
                in_channels=in_shapes[convolution],
                out_channels=kernels,
                kernel_size=kernel_size,
                padding=pad,
                dilation=dilation,
                stride=stride,
            )
            self.model.add_module("conv %i" % (convolution + 1), conv)
            self.model.add_module("ReLU %i" % (convolution + 1), nn.ReLU(inplace=True))
        self.model.add_module("GAP", nn.AdaptiveAvgPool2d(1))
        self.model.add_module("Flatten", nn.Flatten())
        for fully_connect in range(fully_connected):
            if fully_connect == fully_connected - 1:
                self.model.add_module("FC %i" % fully_connect, nn.Linear(kernels, 1))
            else:
                self.model.add_module(
                    "FC %i" % fully_connect, nn.Linear(kernels, kernels)
                )
        self.model.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x):
        out = self.model(x)
        return out
