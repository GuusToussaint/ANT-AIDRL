import torch
import torch.nn as nn
import torch.optim as optim
import random

from ops import binary_indicator, stochastic_binary_indicator, depth_min, depth_inc




class ANT:
    def __init__(self, in_shape, num_classes,
                 new_router, new_transformer, new_solver,
                 soft_decision=True, stochastic=False):
        """ Adaptive Neural Tree (https://arxiv.org/pdf/1807.06699.pdf)

        in_shape gives the input shape (excluding first batch dimension).

        num_classes the number of classes (where 1 is regression).

        new_router/new_transformer must be callables taking an input shape
        and returning a Router/Transformer object.

        new_solver must be a callable taking an input shape and num_classes
        and returning a Solver object.

        soft_decision makes each router a soft decision node. If false
        the router either stochastically or greedily chooses the left or right child.

        """

        self.in_shape = in_shape
        self.num_classes = num_classes

        self.new_router = new_router
        self.new_transformer = new_transformer
        self.new_solver = new_solver

        self.soft_decision = soft_decision
        self.stochastic = stochastic

        self.training = False
        self.root = SolverNode(self, in_shape)

    def fit(self, X, y):
        pass  # TODO

    def predict(self, X, y):
        pass  # TODO


class TreeNode(nn.Module):
    def __init__(self, ant, in_shape, out_shape):
        super().__init__()
        self.ant = ant
        self.unexpanded_depth = None
        self.in_shape = in_shape
        self.out_shape = out_shape

    def fully_expanded(self):
        return self.unexpanded_depth is None

    def expand(self):
        raise NotImplementedError

    def set_frozen(self, frozen, recursive=False):
        raise NotImplementedError



class RouterNode(TreeNode):
    def __init__(self, ant, router, left_child, right_child):
        super().__init__(ant, router.in_shape, router.in_shape)
        self.left_child = left_child
        self.right_child = right_child
        self.unexpanded_depth = depth_inc(depth_min(self.left_child.unexpanded_depth,
                                                   self.right_child.unexpanded_depth))
        self.router = router


    def expand(self, *args, **kwargs):
        ld = self.left_child.unexpanded_depth
        rd = self.right_child.unexpanded_depth
        md = depth_min(ld, rd)
        if md is None:
            raise RuntimeError("expand called on fully expanded tree")

        self.set_frozen(True, recursive=False)
        if ld == md:
            self.right_child.set_frozen(True, recursive=True)
            self.left_child = self.left_child.expand(*args, **kwargs)
        else:
            self.left_child.set_frozen(True, recursive=True)
            self.right_child = self.right_child.expand(*args, **kwargs)

        self.unexpanded_depth = depth_inc(depth_min(self.left_child.unexpanded_depth,
                                                    self.right_child.unexpanded_depth))
        return self


    def forward(self, x):
        p = self.router(x)
        if self.ant.soft_decision or self.ant.training:
            return p*self.left_child(x) + (1 - p)*self.right_child(x)

        if self.ant.stochastic:
            ind = stochastic_binary_indicator(p)
        else:
            ind = binary_indicator(p)

        o = x.new_empty(x.size())
        left_mask = ind > 0.5
        right_mask = ~ind
        o[left_mask] = self.left_child(x[left_mask])
        o[right_mask] = self.right_child(x[right_mask])
        return o


    def set_frozen(self, frozen, recursive=False):
        for param in self.router.parameters():
            param.requires_grad = not frozen

        if recursive:
            self.left_child.set_frozen(frozen, True)
            self.right_child.set_frozen(frozen, True)



class TransformerNode(TreeNode):
    def __init__(self, ant, transformer, child):
        super().__init__(ant, transformer.in_shape, transformer.out_shape)
        self.child = child
        self.unexpanded_depth = depth_inc(self.child.unexpanded_depth)
        self.transformer = transformer


    def expand(self, *args, **kwargs):
        if self.child.unexpanded_depth is None:
            raise RuntimeError("expand called on fully expanded tree")

        self.set_frozen(True, recursive=False)
        self.child = self.child.expand(*args, **kwargs)
        self.unexpanded_depth = depth_inc(self.child.unexpanded_depth)
        return self


    def set_frozen(self, frozen, recursive=False):
        for param in self.transformer.parameters():
            param.requires_grad = not frozen

        if recursive:
            self.child.set_frozen(frozen, True)



class SolverNode(TreeNode):
    def __init__(self, ant, solver):
        super().__init__(ant, solver.in_shape, (1,))
        self.unexpanded_depth = 0
        self.solver = solver

    def expand(self):
        # TODO: test best approach.
        r = random.random()
        if r < 0.25:
            r = self.ant.new_router(self.in_shape)
            s1 = SolverNode(self.ant, self.ant.new_solver(r.out_shape, self.ant.num_classes))
            s2 = SolverNode(self.ant, self.ant.new_solver(r.out_shape, self.ant.num_classes))
            return RouterNode(self.ant, r, s1, s2)
        if r < 0.5:
            t = self.ant.new_transformer(self.in_shape)
            s = SolverNode(self.ant, self.ant.new_solver(t.out_shape, self.ant.num_classes))
            return TransformerNode(self.ant, t, s)

        # No expansion.
        self.unexpanded_depth = None
        return self

