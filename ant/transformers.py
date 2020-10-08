import torch.nn as nn
from math import floor

class Transformer(nn.Module):
    """ Arbitrarily transforms its input, into a fixed output shape. """
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape


class FullyConnected1DTransformer(Transformer):
    def __init__(self, in_shape):
        super().__init__(in_shape, in_shape)
        # assert len(in_shape) == 1

        # TODO make this parametrizable
        self.model = nn.Sequential(
            nn.Linear(in_shape[0], 50),
            nn.ReLU(),
            nn.Linear(50, in_shape[0]),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class Conv2DRelu(Transformer):
    def __init__(self, in_shape, convolutions=1, kernels=40, kernel_size=5, pad=0, dilation=1, stride=1):
        # TODO: only works with square images
        prev_size = size = in_shape[1]
        in_shapes = [in_shape[0]]
        for convolution in range(convolutions):
            size = floor(((prev_size + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
            if convolution > 0:
                in_shapes.append(kernels)
            prev_size = size
        out_shape = [kernels, size, size]
        super().__init__(in_shape, out_shape)

        self.model = nn.Sequential()
        for convolution in range(convolutions):
            conv = nn.Conv2d(in_channels=in_shapes[convolution], out_channels=kernels, kernel_size=kernel_size)
            self.model.add_module("conv %i" % (convolution + 1), conv)
        self.model.add_module("ReLU", nn.ReLU())

    def forward(self, x):
        out = self.model(x)
        return out
