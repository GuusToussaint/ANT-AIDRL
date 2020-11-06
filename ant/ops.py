from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


# https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/datamodules/sklearn_datamodule.py#L19-L63
class SklearnDataset(Dataset):
    """
    Mapping between numpy (or sklearn) datasets to PyTorch datasets.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, X_transform: Any = None, y_transform: Any = None):
        """
        Args:
            X: Numpy ndarray
            y: Numpy ndarray
            X_transform: Any transform that works with Numpy arrays
            y_transform: Any transform that works with Numpy arrays
        """
        super().__init__()
        self.X = X
        self.Y = y
        self.X_transform = X_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.X_transform:
            x = self.X_transform(x)

        if self.y_transform:
            y = self.y_transform(y)

        return x, y

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

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

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
    new_optimizer,
    max_epochs,
    *,
    device=None,
    val_loader=None,
    patience=None,
    verbose=False,
    refinement=False,
    lr_factor=0.1
):
    if device:
        model.to(device)

    if patience is not None and val_loader is None:
        raise ValueError("patience require a validation set")

    last_val_loss = float("inf")
    no_improvement_epochs = 0

    optimizer = new_optimizer(model.parameters())
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
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
            n += len(data)

        epoch_loss /= n

        if val_loader and (verbose or patience is not None):
            val_loss = eval(model, val_loader, loss_function, device=device)
        else:
            val_loss = None

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

        if epoch+1%50 == 0 and refinement:
            print("changing learning rate")
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * lr_factor

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
            n += len(data)

    return total / n
