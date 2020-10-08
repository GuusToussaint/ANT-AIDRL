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
    if a is None and b is None: return None
    if a is None: return b
    if b is None: return a
    return min(a, b)

def depth_inc(x):
    """ Returns x + 1, respecting None as positive infinity. """
    return None if x is None else x + 1



# Torch generic train/eval functions.
def train(model, train_loader, loss_function, new_optimizer, max_epochs, *,
          device=None, val_loader=None, patience=None, verbose=False):
    if device:
        model.to(device)

    if patience is not None and val_loader is None:
        raise ValueError("patience require a validation set")


    last_val_loss = float('inf')
    no_improvement_epochs = 0

    optimizer = new_optimizer(model.parameters())
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            if device:
                inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        
        if val_loader and (verbose or patience is not None):
            val_loss = eval(model, val_loader, loss_function, device=device)
            val_loss /= len(val_loader)
        else:
            val_loss = None
        
        if verbose:
            out = "Epoch {}, train loss {:.5}".format(epoch + 1, epoch_loss)
            if val_loss is not None:
                out += ", val loss {:.5}".format(val_loss)
            print(out)

        if patience is not None:
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
    for i, data in enumerate(data_loader):
        inputs, labels = data
        if device:
            inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        total += loss_function(outputs, labels)
    
    return total