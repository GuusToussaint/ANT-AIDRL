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

        assert len(in_shape) == 1

        self.model = nn.Sequential(
            nn.Linear(in_shape[0], in_shape[0]),
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


        # Convolutional layers.
        current_shape = in_shape
        modules = []
        for _ in range(convolutions):            
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

        if downsample:
            if current_shape[0] >= 2 and current_shape[1] >= 2:
                modules.append(nn.MaxPool2d(kernel_size=(2, 2)))
            else:
                print("skipping max pooling due to the size")
        super().__init__(in_shape, current_shape)
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)
