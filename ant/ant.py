import torch
import torch.nn as nn
import torch.optim as optim
import random

from .ops import binary_indicator, stochastic_binary_indicator, depth_min, depth_inc, Stack




class ANT:
    def __init__(self, in_shape, num_classes,
                 new_router, new_transformer, new_solver,
                 new_optimizer,
                 soft_decision=True, stochastic=False):
        """ Adaptive Neural Tree (https://arxiv.org/pdf/1807.06699.pdf)

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

        """

        self.in_shape = in_shape
        self.num_classes = num_classes

        self.new_router = new_router
        self.new_transformer = new_transformer
        self.new_solver = new_solver
        self.new_optimizer = new_optimizer

        self.soft_decision = soft_decision
        self.stochastic = stochastic

        self.training = False


        if num_classes == 1:
            self.loss_function = nn.MSELoss()
        else:
            nll_loss = nn.NLLLoss()
            self.loss_function = lambda pred, target: nll_loss(torch.log(pred), target)

    def do_train(self, train_loader, val_loader, expand_epochs, final_epochs, *, device="cpu", verbose=True):
        self.training = True

        try:
            if verbose:
                print("Starting tree build process.")

            # Create initial leaf and train it.
            self.root = SolverNode(self, self.new_solver(self.in_shape, self.num_classes))
            self.root.do_train(train_loader, expand_epochs, device=device, verbose=verbose)

            # Expand while possible.
            while not self.root.fully_expanded():
                self.root = self.root.expand(train_loader, val_loader, expand_epochs, device=device, verbose=verbose)
            
            # Final refinement.
            if verbose:
                print("Starting final refinement.")
            self.root.set_frozen(False, recursive=True)
            self.root.do_train(train_loader, final_epochs, device=device, verbose=verbose)

        finally:
            self.training = False
        


    def fit(self, X, y):
        raise NotImplementedError   # TODO, sklearn interface

    def predict(self, X, y):
        raise NotImplementedError   # TODO, sklearn interface



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

    def do_train(self, train_loader, epochs, *, device, verbose, multi_head=False):
        self.to(device)
        self.train()
        optimizer = self.ant.new_optimizer(self.parameters())
        running_loss = 0.0
        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                if multi_head:
                    loss = sum(self.ant.loss_function(output_head, labels)
                               for output_head in outputs)
                else:
                    loss = self.ant.loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # if verbose:
                #     # Print every 100 mini-batches.
                #     if i % 100 == 99:
                #         print("[{}, {:5}] loss: {:.3}".format(epoch + 1, i + 1, running_loss / 100))
                #         running_loss = 0.0

    def do_eval(self, val_loader, *, device, multi_head=False):
        self.to(device)
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = self(inputs)
                if multi_head:
                    loss = torch.tensor([self.ant.loss_function(output_head, labels)
                                         for output_head in outputs])
                else:
                    loss = self.ant.loss_function(outputs, labels)
                total_loss += loss
        return total_loss
        




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

        raise NotImplementedError
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

    def expand(self, train_loader, val_loader, expand_epochs, *, device, verbose):
        if verbose:
            print("Attempting to expand leaf node.")

        # Mark as expanded.
        self.unexpanded_depth = None

        # Use pre-trained existing leaf.
        leaf_candidate = self.solver

        # Create transformer.
        # FIXME: should this new leaf also be initialized with the old leaf weights?
        t = self.ant.new_transformer(self.in_shape)
        s = SolverNode(self.ant, self.ant.new_solver(t.out_shape, self.ant.num_classes))
        transformer_candidate = TransformerNode(self.ant, t, s)

        # Create router.
        # FIXME: should these new leafs also be initialized with the old leaf weights?
        r = self.ant.new_router(self.in_shape)
        s1 = SolverNode(self.ant, self.ant.new_solver(r.out_shape, self.ant.num_classes))
        s2 = SolverNode(self.ant, self.ant.new_solver(r.out_shape, self.ant.num_classes))
        router_candidate = RouterNode(self.ant, r, s1, s2)

        try:
            # Monkey-patch this nodes solver.
            self.solver = Stack(leaf_candidate, transformer_candidate, router_candidate)
            self.ant.root.do_train(train_loader, expand_epochs, device=device, verbose=verbose, multi_head=True)
            val_losses = self.ant.root.do_eval(val_loader, device=device, multi_head=True)
        finally:
            # Restore self.
            self.solver = leaf_candidate

        # Use best node.
        best = val_losses.argmin().item()
        if verbose:
            print("Best choice was {} with respective losses {} for leaf/transformer/router.".format(["leaf", "transformer", "router"][best], val_losses))
        return [self, transformer_candidate, router_candidate][best]

    def forward(self, x):
        return self.solver(x)

    def set_frozen(self, frozen, recursive=False):
        for param in self.solver.parameters():
            param.requires_grad = not frozen