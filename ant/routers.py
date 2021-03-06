import torch
import torch.nn as nn
import numpy as np

from . import ops


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
        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(np.prod(in_shape), 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Conv2DGAPFCSigmoidRouter(Router):
    def __init__(
        self, in_shape, *, convolutions, kernel_size, kernels, fc_layers, fc_reduction=2
    ):
        super().__init__(in_shape)

        # Convolutional layers.
        current_shape = in_shape
        modules = []
        for i in range(convolutions):

            new_shape = ops.conv_output_shape(
                tuple(current_shape[-2:]), kernel_size=kernel_size
            )

            if new_shape [0] >= kernel_size or new_shape[1] >= kernel_size:
                conv = nn.Conv2d(
                    in_channels=current_shape[0],
                    out_channels=kernels,
                    kernel_size=kernel_size,
                )
                modules.append(conv)
                modules.append(nn.ReLU())

                current_shape = (kernels,) + new_shape
            else: 
                print("skipping this convolution due to the size")
                break

        # Global average pooling.
        modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(nn.Flatten())

        # Fully connected layers.
        neurons = current_shape[0]
        for i in range(fc_layers):
            if i != fc_layers - 1:
                modules.append(nn.Linear(neurons, 1 + neurons // fc_reduction))
                modules.append(nn.ReLU())
                neurons = 1 + neurons // fc_reduction
            else:
                modules.append(nn.Linear(neurons, 1))
        modules.append(nn.Sigmoid())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)
