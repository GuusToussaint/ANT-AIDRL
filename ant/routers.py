import torch.nn as nn


class Router(nn.Module):
    """ Given it's input outputs a scalar representing a probability
        of using the left subtree in the ANT. """
    def __init__(self, in_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = in_shape


class FullyConnectedSigmoidRouter(Router):
    def __init__(self, in_shape):
        super().__init__(in_shape)
        assert len(in_shape) == 1
        self.model = nn.Sequential(nn.Linear(in_shape[0], 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)
