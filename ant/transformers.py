import torch.nn as nn


class Transformer(nn.Module):
    """ Arbitrarily transforms its input, into a fixed output shape. """
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape


class FullyConnected1DTransformer(Transformer):
    def __init__(self, in_shape):
        super().__init__(in_shape, in_shape)
        self.fc = nn.Linear(in_shape, in_shape)

    def forward(self, x):
        return self.fc(x)
