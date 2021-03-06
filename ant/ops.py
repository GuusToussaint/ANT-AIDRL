from typing import Any
import torch
from torch._C import dtype
import torch.nn as nn
import numpy as np
import pickle

# References: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
# https://github.com/rtanno21609/AdaptiveNeuralTrees/blob/master/ops.py
class BinaryIndicator(torch.autograd.Function):
    """Indicator function that turns x < 0.5 to 0 and x >= 0.5 to 1.
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
    """Stochastic version of BinaryIndicator. Returns 1 with probability x, 0 otherwise.
    Gradient is passed straight through.
    """

    @staticmethod
    def forward(ctx, x):
        r = x.new_empty(x.size()).uniform_(0, 1)
        return torch.abs(torch.round(x - r + 0.5))

    @staticmethod
    def backward(ctx, g):
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
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def depth_inc(x):
    """ Returns x + 1, respecting None as positive infinity. """
    return None if x is None else x + 1


# https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions.
    Takes a tuple of (h,w) and returns a tuple of (h,w).
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = int(
        ((h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) / stride[0])
        + 1
    )
    w = int(
        ((h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) / stride[1])
        + 1
    )


    return h, w


def format_loss(loss):
    """ Formats a loss value. """
    if isinstance(loss, float):
        return f"{loss:.5}"
    if isinstance(loss, torch.Tensor):
        if len(loss.shape) == 0:
            return f"{loss.item():.5}"
        if len(loss.shape) == 1:
            return "[" + ", ".join(f"{x:.5}" for x in loss) + "]"
    return str(loss)


# Torch generic train/eval functions.
def train(
    model,
    train_loader,
    loss_function,
    optimizer,
    max_epochs,
    *,
    batch_scheduler=None,
    epoch_scheduler=None,
    device=None,
    val_loader=None,
    patience=None,
    verbose=False,
    scheduler=None,
):
    
    if verbose:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"started training loop for model with {params} trainable parameters")

    if device:
        model.to(device)

    if patience is not None and val_loader is None:
        raise ValueError("patience require a validation set")

    last_val_loss = float("inf")
    no_improvement_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n = 0

        for i, data in enumerate(train_loader):
            # One batch.
            inputs, labels = data
            if torch.isnan(torch.sum(inputs)) or torch.isinf(torch.sum(inputs)):
                # FIXME: remove this.
                print("INPUTS ARE NAN")
                break

            if device:
                inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            epoch_loss += loss
            if len(loss.shape) == 1:
                loss = torch.sum(loss)
            loss.backward()
            optimizer.step()
            if batch_scheduler is not None:
                batch_scheduler()
            n += labels.size(0)

        epoch_loss /= n

        if val_loader and (verbose or patience is not None or epoch_scheduler is not None):
            val_loss = eval(model, val_loader, loss_function, device=device)
        else:
            val_loss = None
        
        if epoch_scheduler is not None:
            epoch_scheduler(val_loss)

        if verbose:
            out = "Epoch {}, train loss {}".format(epoch + 1, format_loss(epoch_loss))
            if val_loss is not None:
                out += ", val loss {}".format(format_loss(val_loss))
            print(out)

        if patience is not None:
            if len(val_loss.shape) == 1:
                val_loss = torch.sum(val_loss)
            if val_loss >= last_val_loss:
                no_improvement_epochs += 1
            if no_improvement_epochs > patience:
                break
        
        last_val_loss = val_loss
        


def eval(model, data_loader, loss_function, *, device=None):
    if device:
        model.to(device)

    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            if device:
                inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total += loss
            n += labels.size(0)

    return total / n
