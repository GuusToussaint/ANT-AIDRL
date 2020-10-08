import torch
import torch.nn as nn
import torch.optim as optim
import random

from . import ops
from .ops import (
    binary_indicator,
    stochastic_binary_indicator,
    depth_min,
    depth_inc,
    Stack,
)


class ANT:
    def __init__(
        self,
        in_shape,
        num_classes,
        new_router,
        new_transformer,
        new_solver,
        new_optimizer,
        soft_decision=True,
        stochastic=False,
        router_inherit=True,
        transformer_inherit=False,
    ):
        """Adaptive Neural Tree (https://arxiv.org/pdf/1807.06699.pdf)

        in_shape gives the input shape (excluding first batch dimension).

        num_classes the number of classes (where 1 is regression).

        new_router/new_transformer must be callables taking an input shape
        and returning a Router/Transformer object.

        new_solver must be a callable taking an input shape and num_classes
        and returning a Solver object.

        new_optimizer must be a callable taking an iterable of params and
        returning a torch.optim.Optimizer object.

        soft_decision makes each router a soft decision node. If false
        the router either stochastically or greedily chooses the left or right child.

        router_inherit/transformer_inherit decides whether the new leaf nodes
        generated by replacing a leaf with a router/transformer inherit the
        weights of the old solver.

        """

        self.in_shape = in_shape
        self.num_classes = num_classes

        self.new_router = new_router
        self.new_transformer = new_transformer
        self.new_solver = new_solver
        self.new_optimizer = new_optimizer

        self.soft_decision = soft_decision
        self.stochastic = stochastic

        self.router_inherit = router_inherit
        self.transformer_inherit = transformer_inherit

        self.training = False

        if num_classes == 1:
            self.loss_function = nn.MSELoss()
        else:
            nll_loss = nn.NLLLoss()
            self.loss_function = lambda pred, target: nll_loss(torch.log(pred), target)

    def do_train(
        self,
        train_loader,
        val_loader,
        max_expand_epochs,
        max_final_epochs,
        *,
        device="cpu",
        verbose=True
    ):
        self.training = True

        try:
            if verbose:
                print("Starting tree build process.")

            # Create initial leaf and train it.
            self.root = SolverNode(
                self, self.new_solver(self.in_shape, self.num_classes)
            )
            ops.train(
                self.root,
                train_loader,
                self.loss_function,
                self.new_optimizer,
                max_expand_epochs,
                device=device,
                verbose=verbose,
            )

            # Expand while possible.
            while not self.root.fully_expanded():
                self.root = self.root.expand(
                    train_loader,
                    val_loader,
                    max_expand_epochs,
                    device=device,
                    verbose=verbose,
                )

            # Final refinement.
            if verbose:
                print("Starting final refinement.")
            self.root.set_frozen(False, recursive=True)
            ops.train(
                self.root,
                train_loader,
                self.loss_function,
                self.new_optimizer,
                max_expand_epochs,
                device=device,
                verbose=verbose,
            )

        finally:
            self.training = False

    def get_tree_composition(self):
        """ Returns (num_router, num_transformer, num_solver). """
        return self.root.get_tree_composition()

    def fit(self, X, y):
        raise NotImplementedError  # TODO, sklearn interface

    def predict(self, X, y):
        raise NotImplementedError  # TODO, sklearn interface


class TreeNode(nn.Module):
    def __init__(self, ant, in_shape, out_shape):
        super().__init__()
        self.ant = ant
        self.unexpanded_depth = None
        self.in_shape = in_shape
        self.out_shape = out_shape

    def fully_expanded(self):
        return self.unexpanded_depth is None

    def get_tree_composition(self):
        raise NotImplementedError

    def expand(self):
        raise NotImplementedError

    def set_frozen(self, frozen, recursive=False):
        raise NotImplementedError


class RouterNode(TreeNode):
    def __init__(self, ant, router, left_child, right_child):
        super().__init__(ant, router.in_shape, router.in_shape)
        self.left_child = left_child
        self.right_child = right_child
        self.unexpanded_depth = depth_inc(
            depth_min(
                self.left_child.unexpanded_depth, self.right_child.unexpanded_depth
            )
        )
        self.router = router

    def get_tree_composition(self):
        num_router, num_transformer, num_solver = self.left_child.get_tree_composition()
        return (num_router + r[0] + 1, num_transformer + r[1], num_solver + r[2])

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

        self.unexpanded_depth = depth_inc(
            depth_min(
                self.left_child.unexpanded_depth, self.right_child.unexpanded_depth
            )
        )
        return self

    def forward(self, x):
        p = self.router(x)
        if self.ant.soft_decision or self.ant.training:
            return p * self.left_child(x) + (1 - p) * self.right_child(x)

        raise NotImplementedError

        # FIXME: implement single-path routing?
        # if self.ant.stochastic:
        #     ind = stochastic_binary_indicator(p)
        # else:
        #     ind = binary_indicator(p)

        # o = x.new_empty(x.size())
        # left_mask = ind > 0.5
        # right_mask = ~ind
        # o[left_mask] = self.left_child(x[left_mask])
        # o[right_mask] = self.right_child(x[right_mask])
        # return o

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

    def get_tree_composition(self):
        num_router, num_transformer, num_solver = self.child.get_tree_composition()
        return (num_router, num_transformer + 1, num_solver)

    def expand(self, *args, **kwargs):
        if self.child.unexpanded_depth is None:
            raise RuntimeError("expand called on fully expanded tree")

        self.set_frozen(True, recursive=False)
        self.child = self.child.expand(*args, **kwargs)
        self.unexpanded_depth = depth_inc(self.child.unexpanded_depth)
        return self

    def forward(self, x):
        return self.child(self.transformer(x))

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

    def get_tree_composition(self):
        return (0, 0, 1)

    def expand(self, train_loader, val_loader, max_expand_epochs, *, device, verbose):
        if verbose:
            print("Attempting to expand leaf node.")

        # Mark leaf as expanded.
        self.unexpanded_depth = None

        # Use pre-trained existing leaf.
        leaf_candidate = self.solver

        # Create transformer.
        t = self.ant.new_transformer(self.in_shape)
        s = SolverNode(self.ant, self.ant.new_solver(t.out_shape, self.ant.num_classes))
        transformer_candidate = TransformerNode(self.ant, t, s)
        if self.ant.transformer_inherit:
            s.solver.load_state_dict(leaf_candidate.state_dict())

        # Create router.
        r = self.ant.new_router(self.in_shape)
        s1 = SolverNode(
            self.ant, self.ant.new_solver(r.out_shape, self.ant.num_classes)
        )
        s2 = SolverNode(
            self.ant, self.ant.new_solver(r.out_shape, self.ant.num_classes)
        )
        router_candidate = RouterNode(self.ant, r, s1, s2)
        if self.ant.router_inherit:
            s1.solver.load_state_dict(leaf_candidate.state_dict())
            s2.solver.load_state_dict(leaf_candidate.state_dict())

        multi_head_loss = lambda outputs, labels: sum(
            self.ant.loss_function(output_head, labels) for output_head in outputs
        )

        try:
            # Monkey-patch this nodes' solver.
            self.solver = Stack(leaf_candidate, transformer_candidate, router_candidate)
            ops.train(
                self.ant.root,
                train_loader,
                multi_head_loss,
                self.ant.new_optimizer,
                max_expand_epochs,
                val_loader=val_loader,
                device=device,
                verbose=verbose,
            )
            val_losses = ops.eval(
                self.ant.root, val_loader, multi_head_loss, device=device
            )
        finally:
            # Restore self.
            self.solver = leaf_candidate

        # Use best node.
        best = val_losses.argmin().item()
        if verbose:
            print(
                "Best choice was {} with respective losses {} for leaf/transformer/router.".format(
                    ["leaf", "transformer", "router"][best], val_losses
                )
            )
        return [self, transformer_candidate, router_candidate][best]

    def forward(self, x):
        return self.solver(x)

    def set_frozen(self, frozen, recursive=False):
        for param in self.solver.parameters():
            param.requires_grad = not frozen
