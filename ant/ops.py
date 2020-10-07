import torch
import torch.nn as nn

# References: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
# https://github.com/rtanno21609/AdaptiveNeuralTrees/blob/master/ops.py
class BinaryIndicator(torch.autograd.Function):
    """ Indicator function that turns x < 0.5 to 0 and x >= 0.5 to 1.
        Gradient is passed straight through.
    """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g
binary_indicator = BinaryIndicator.apply


class StochasticBinaryIndicator(torch.autograd.Function):
    """ Stochastic version of BinaryIndicator. Returns 1 with probability x, 0 otherwise.
        Gradient is passed straight through.
    """
    @staticmethod
    def forward(self, x):
        r = x.new_empty(x.size()).uniform_(0, 1)
        return torch.abs(torch.round(x - r + 0.5))

    @staticmethod
    def backward(self, g):
        return g
stochastic_binary_indicator = StochasticBinaryIndicator.apply


class Stack(nn.Module):
    """ Stacks torch modules in the given axis. """
    def __init__(self, *modules, dim=0):
        super().__init__()
        self.inner = nn.ModuleList(modules)
        self.dim = dim

    def forward(self, x):
        ys = [f(x) for f in self.inner]
        return torch.stack(ys, dim=self.dim)


def depth_min(a, b):
    """ Returns the minimum of a and b, respecting None as positive infinity. """
    if a is None and b is None: return None
    if a is None: return b
    if b is None: return a
    return min(a, b)

def depth_inc(x):
    """ Returns x + 1, respecting None as positive infinity. """
    return None if x is None else x + 1

