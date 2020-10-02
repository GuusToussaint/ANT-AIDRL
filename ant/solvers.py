import torch.nn

class Solver(nn.Module):
    """ Given its input outputs its prediction for each class. When num_classes = 1,
        assume regression, otherwise output log probabilities for each class. """

    def __init__(self, in_shape, num_classes):
        self.in_shape = in_shape
        self.num_classes = num_classes



class Linear1D(Solver):
    def __init__(self, in_shape, num_classes):
        super().__init__(in_shape, num_classes)
        if num_classes == 1:
            self.model = nn.Linear(in_shape, num_classes)
        else:
            self.model = nn.Sequential(nn.Linear(in_shape, num_classes), nn.LogSoftMax())

    def forward(self, x):
        return self.model(x)
