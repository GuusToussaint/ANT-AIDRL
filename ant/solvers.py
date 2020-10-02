import torch.nn as nn

class Solver(nn.Module):
    """ Given its input outputs its prediction for each class. When num_classes = 1,
        assume regression, otherwise output log probabilities for each class. """

    def __init__(self, in_shape, num_classes):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = (1,)
        self.num_classes = num_classes



class Linear1DSolver(Solver):
    def __init__(self, in_shape, num_classes):
        super().__init__(in_shape, num_classes)
        assert len(in_shape) == 1

        if num_classes == 1:
            self.model = nn.Linear(in_shape[0], num_classes)
        else:
            self.model = nn.Sequential(nn.Linear(in_shape[0], num_classes), nn.LogSoftmax())

    def forward(self, x):
        return self.model(x)
