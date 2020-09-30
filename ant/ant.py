import torch
import torch.nn as nn
import random

from ops import binary_indicator, stochastic_binary_indicator, depth_min, depth_inc




class ANT:
    def __init__(self):
        self.root = Solver(self)
        self.new_solver_model = lambda in_shape: nn.Linear(in_shape, 1)
        self.new_router_model = lambda in_shape: nn.Sequential(nn.Linear(in_shape, 1), nn.Sigmoid())
        self.new_transformer_model = lambda in_shape: nn.Sequential(nn.Linear(in_shape, in_shape), nn.ReLU())

    def fit(self, X, y):
        pass  # TODO

    def predict(self, X, y):
        pass  # TODO


class TreeNode(nn.Module):
    def __init__(self, ant, in_shape):
        super().__init__()
        self.ant = ant
        self.unexpanded_depth = None
        self.in_shape = in_shape

    def fully_expanded(self):
        return self.unexpanded_depth is None

    def bfs_unexpanded_depth(self):
        """ Returns the depth of the first unexpanded node in BFS order,
            or None if there aren't any unexpanded nodes. """
        raise NotImplementedError

    def expand(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError



class Router(TreeNode):
    def __init__(self, ant, in_shape, left_child, right_child):
        super().__init__(ant, in_shape)
        self.left_child = left_child
        self.right_child = right_child
        self.unexpanded_depth = depth_inc(depth_min(self.left_child.unexpanded_depth,
                                                   self.right_child.unexpanded_depth))
        self.model = ant.new_router_model(in_shape)


    def expand(self, *args, **kwargs):
        ld = self.left_child.unexpanded_depth
        rd = self.right_child.unexpanded_depth
        md = depth_min(ld, rd)
        if md is not None:
            if ld == md:
                self.left_child = self.left_child.expand(*args, **kwargs)
            else:
                self.right_child = self.right_child.expand(*args, **kwargs)

            self.unexpanded_depth = depth_inc(depth_min(self.left_child.unexpanded_depth,
                                                        self.right_child.unexpanded_depth))

        return self

    def forward(self, x):
        p = self.model(x)
        if ant.soft_decision:
            return p*self.left_child(x) + (1 - p)*self.right_child(x)

        if ant.stochastic:
            ind = stochastic_binary_indicator(p)
        else:
            ind = binary_indicator(p)

        o = x.new_empty(x.size())
        left_mask = ind > 0.5
        right_mask = ~ind
        o[left_mask] = self.left_child(x[left_mask])
        o[right_mask] = self.right_child(x[right_mask])
        return o








class Transformer(TreeNode):
    def __init__(self, ant, in_shape, child):
        super().__init__(ant, in_shape)
        self.child = child
        self.unexpanded_depth = depth_inc(self.child.unexpanded_depth)
        self.model = ant.new_transformer_model(in_shape)


    def expand(self, *args, **kwargs):
        if self.child.unexpanded_depth is not None:
            self.child = self.child.expand(*args, **kwargs)
            self.unexpanded_depth = depth_inc(self.child.unexpanded_depth)

        return self


class Solver(TreeNode):
    def __init__(self, ant, in_shape):
        super().__init__(ant, in_shape)
        self.unexpanded_depth = 0
        self.model = ant.new_solver_model(in_shape)

    def expand(self):
        # TODO: test best approach.
        r = random.random()
        if r < 0.25:
            return Split(Solver(self.ant), Solver(self.ant))
        if r < 0.5:
            return Transformer(self.ant, Solver(self.ant))

        self.unexpanded_depth = None
        return self

