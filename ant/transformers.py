import torch.nn as nn


class Transformer(nn.Module):
    """ Arbitrarily transforms its input, into a fixed output shape. """
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape


class FullyConnected1DTransformer(Transformer):
    def __init__(self, in_shape):
        super().__init__(in_shape, in_shape)
        assert len(in_shape) == 1

        # TODO make this parametrizable
        self.model = nn.Sequential(
            nn.Linear(in_shape[0], 50),
            nn.ReLU(),
            nn.Linear(50, in_shape[0]),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)
