import torch.nn as nn
from math import floor
from . import ops


class Transformer(nn.Module):
    """ Arbitrarily transforms its input, into a fixed output shape. """

    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape


class IdentityTransformer(Transformer):
    def __init__(self, in_shape):
        super().__init__(in_shape, in_shape)

    def forward(self, x):
        return x


class FullyConnectedTransformer(Transformer):
    def __init__(self, in_shape, fc_reduction=2):
        super().__init__(
            in_shape,
            in_shape,
        )
        # assert len(in_shape) == 1

        self.model = nn.Sequential(
            nn.Linear(in_shape[0], in_shape[1] // fc_reduction),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Conv2DRelu(Transformer):
    def __init__(
        self,
        in_shape,
        prev_transformers,
        convolutions=1,
        kernels=40,
        kernel_size=5,
        down_sample_freq=1,
    ):
        downsample = False

        # Getting the ouput shape
        if (prev_transformers + 1) % down_sample_freq == 0:
            downsample = True

        _old_kernel_size = kernel_size
        kernel_size = min((kernel_size,) + in_shape[1:3])
        if _old_kernel_size != kernel_size:
            downsample = False

        current_shape = tuple(in_shape[1:3])
        for _ in range(convolutions):
            current_shape = ops.conv_output_shape(
                current_shape, kernel_size=kernel_size
            )
        if downsample:
            current_shape = (
                floor(((i + 2 * 0 - 1 * (2 - 1)) / 2) + 1) for i in current_shape
            )
        out_shape = (kernels, *current_shape)

        super().__init__(in_shape, out_shape)

        # Convolutional layers.
        shape = in_shape
        modules = []
        for i in range(convolutions):
            conv = nn.Conv2d(
                in_channels=shape[0],
                out_channels=kernels,
                kernel_size=kernel_size,
            )
            modules.append(conv)
            shape = (kernels,) + ops.conv_output_shape(tuple(shape[1:3]), kernel_size)
            modules.append(nn.ReLU())
        if downsample:
            modules.append(nn.MaxPool2d(kernel_size=(2, 2)))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)
