from numpy.core.fromnumeric import shape
import torch
from torch._C import dtype
import torch.nn as nn
from graphviz import Digraph
from torch.nn.modules.loss import MSELoss
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import Tensor
import gc
import numpy as np
import random
import pickle
import time

from . import ops
from .ops import (
    # binary_indicator,
    # stochastic_binary_indicator,
    depth_min,
    depth_inc,
    Stack,
)


class ANT:
    def __init__(
        self,
        ant_name,
        in_shape,
        num_classes,
        new_router,
        new_transformer,
        new_solver,
        new_optimizer,
        new_batch_scheduler=lambda opt: None,
        new_epoch_scheduler=lambda opt: None,
        soft_decision=True,
        stochastic=False,
        router_inherit=True,
        transformer_inherit=False,
        growth_patience=5,
        regression=False,
        use_router=True,
        max_depth=10
    ):
        """Adaptive Neural Tree (https://arxiv.org/pdf/1807.06699.pdf)

        ant_name defines a name for the ANT

        in_shape gives the input shape (excluding first batch dimension).

        num_classes the number of classes

        new_router/new_transformer must be callables taking an input shape
        and returning a Router/Transformer object.

        new_solver must be a callable taking an input shape and num_classes
        and returning a Solver object.

        new_optimizer must be a callable taking an iterable of params and
        returning a torch.optim.Optimizer object.

        new_batch_scheduler must be a callable taking an optimizer returning a
        function that gets called without any arguments after every batch.

        new_epoch_scheduler must be a callable taking an optimizer returning a
        function that gets called with the validation loss after every epoch.

        soft_decision makes each router a soft decision node. If false the
        router either stochastically or greedily chooses the left or right
        child.

        router_inherit/transformer_inherit decides whether the new leaf nodes
        generated by replacing a leaf with a router/transformer inherit the
        weights of the old solver.

        growth_patience determines the max number of iterations without progress
        during growth phase training.

        regression is set to True when a regression task is required, else
        it is set to False

        use_router is set to True when the building of a tree should include 
        routers, else (when we want to mimic a CNN) it is set to False and 
        no routers are placed in the tree building process

        max_depth is the maximal depth a tree should reach
        """

        self.in_shape = in_shape
        self.num_classes = num_classes

        self.new_router = new_router
        self.new_transformer = new_transformer
        self.new_solver = new_solver
        self.new_optimizer = new_optimizer
        self.new_batch_scheduler = new_batch_scheduler
        self.new_epoch_scheduler = new_epoch_scheduler

        self.soft_decision = soft_decision
        self.stochastic = stochastic

        self.router_inherit = router_inherit
        self.transformer_inherit = transformer_inherit

        self.growth_patience = growth_patience

        self.training = False
        self.regression = regression
        self.use_router = use_router
        self.max_depth = max_depth

        self.ant_name = ant_name
        self.best_val_loss = float('inf')

        if self.regression:
            self.loss_function = nn.MSELoss(reduction="sum")
        else:
            self.loss_function = nn.NLLLoss(reduction="sum")

    def do_train(
        self,
        train_loader,
        val_loader,
        max_expand_epochs,
        max_final_epochs,
        *,
        max_expansions=float('inf'),
        device="cpu",
        verbose=True,
        track_history=True,
        output_graphviz=None,
    ):
        self.training = True
        training_start = time.time()

        try:
            if verbose:
                print("Starting tree build process.")

            hist = {
                "training_start": training_start,
                "growth_val_losses": [],
                "growth_endtimes": [],
                "refinement_val_losses": [],
                "refinement_endtimes": [],
            }

            # Create initial leaf and train it.
            self.root = SolverNode(
                self, self.new_solver(self.in_shape, self.num_classes), 0, 0
            )
            optimizer = self.new_optimizer(self.root.parameters())
            batch_scheduler = self.new_batch_scheduler(optimizer)
            epoch_scheduler = self.new_epoch_scheduler(optimizer)
            def hist_epoch_scheduler(val_loss):
                hist["growth_val_losses"].append(val_loss)
                hist["growth_endtimes"].append(time.time() - training_start)
                if epoch_scheduler is not None:
                    epoch_scheduler(val_loss)
            ops.train(
                self.root,
                train_loader,
                self.loss_function,
                optimizer,
                max_expand_epochs,
                batch_scheduler=batch_scheduler,
                epoch_scheduler=hist_epoch_scheduler if track_history else epoch_scheduler,
                val_loader=val_loader,
                device=device,
                verbose=verbose,
                patience=self.growth_patience,
            )

            if output_graphviz:
                self.create_graphviz(output_graphviz)

            # Expand while possible.
            num_expansions = 0
            while not self.root.fully_expanded() and num_expansions < max_expansions:
                try:
                    old = self.state_dict()

                    self.root = self.root.expand(
                        train_loader,
                        val_loader,
                        max_expand_epochs,
                        device=device,
                        verbose=verbose,
                        hist=hist if track_history else None,
                    )
                    num_expansions += 1

                    if output_graphviz:
                        self.create_graphviz(output_graphviz)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        gc.collect()
                        torch.cuda.empty_cache()
                        self.load_state_dict(old)
                        print(
                            "Recovered from out-of-memory error, stopping expansion process."
                        )
                        break
                    else:
                        raise
                except:
                    raise

            # Final refinement.
            if verbose:
                print("Starting final refinement.")

            self.root.set_frozen(False, recursive=True)
            optimizer = self.new_optimizer(self.root.parameters())
            batch_scheduler = self.new_batch_scheduler(optimizer)
            epoch_scheduler = self.new_epoch_scheduler(optimizer)
            def hist_epoch_scheduler(val_loss):
                hist["refinement_val_losses"].append(val_loss)
                hist["refinement_endtimes"].append(time.time() - training_start)
                if val_loss < self.best_val_loss:
                    pickle.dump(self.state_dict(),  open(f"{self.ant_name}-state-dict.p", "wb"))
                    self.best_val_loss = val_loss
                    if verbose:
                        print('Storing new best ANT')
                if epoch_scheduler is not None:
                    epoch_scheduler(val_loss)
            ops.train(
                self.root,
                train_loader,
                self.loss_function,
                optimizer,
                max_final_epochs,
                batch_scheduler=batch_scheduler,
                epoch_scheduler=hist_epoch_scheduler if track_history else epoch_scheduler,
                val_loader=val_loader,
                device=device,
                verbose=verbose,
            )

            return hist

        finally:
            self.training = False

    def create_graphviz(self, fname):
        dot = Digraph(comment="Tree structure of trained ANT")

        nodes = []

        def create_tree(node):
            node_id = str(len(nodes))
            nodes.append(node)

            label = node["kind"]
            label += f' {node["in_shape"]} '
            if node["kind"] == "transformer":
                label += f'- {node["transformer_count"]}'
            dot.node(node_id, label)

            if node["kind"] == "router":
                dot.edge(node_id, create_tree(node["left_child"]))
                dot.edge(node_id, create_tree(node["right_child"]))
            elif node["kind"] == "transformer":
                dot.edge(node_id, create_tree(node["child"]))

            return node_id

        create_tree(self.state_dict()["tree"])
        dot.render(fname, view=False)

    def get_tree_composition(self):
        """ Returns (num_router, num_transformer, num_solver). """
        return self.root.get_tree_composition()

    def state_dict(self):
        return {"tree": self.root.tree_state_dict(), "params": self.root.state_dict()}

    def load_state_dict(self, state_dict):
        self.root = TreeNode.load_tree_state_dict(self, state_dict["tree"])
        self.root.load_state_dict(state_dict["params"])

    def fit(
        self,
        trainset,
        valset,
        batch_size=512,
        verbose=True,
        max_expand_epochs=100,
        max_final_epochs=200,
        max_expansions=float('inf'),
        device=None,
        track_history=True,
        output_graphviz=None,
    ):
        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        if verbose:
            print(
                f"Fitting model with {len(trainset)} training instances and {len(valset)} validation instances."
            )

        return self.do_train(
            trainloader,
            valloader,
            max_expand_epochs=max_expand_epochs,
            max_final_epochs=max_final_epochs,
            max_expansions=max_expansions,
            device=device,
            verbose=verbose,
            track_history=track_history,
            output_graphviz=output_graphviz,
        )

    def eval_acc(self, dataset, batch_size=16, device=None):
        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        if self.regression:
            loss_function = MSELoss(reduction="sum")
        else:
            loss_function = lambda outputs, labels: (
                torch.max(outputs.data, 1).indices == labels).sum().item()
        return ops.eval(self.root, loader, loss_function, device=device)

    def eval_loss(self, dataset, loss_function, batch_size=16, device=None):
        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        return ops.eval(self.root, loader, loss_function, device=device)


class TreeNode(nn.Module):
    def __init__(self, ant, in_shape, out_shape, transformer_count):
        super().__init__()
        self.ant = ant
        self.unexpanded_depth = None
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.transformer_count = transformer_count

    def fully_expanded(self):
        return self.unexpanded_depth is None

    def get_tree_composition(self):
        raise NotImplementedError

    def tree_state_dict(self):
        raise NotImplementedError

    @classmethod
    def load_tree_state_dict(cls, ant, tree_state_dict):
        if tree_state_dict["kind"] == "router":
            return RouterNode.load_tree_state_dict(ant, tree_state_dict)
        elif tree_state_dict["kind"] == "transformer":
            return TransformerNode.load_tree_state_dict(ant, tree_state_dict)
        elif tree_state_dict["kind"] == "solver":
            return SolverNode.load_tree_state_dict(ant, tree_state_dict)
        else:
            raise RuntimeError("Unknown tree state kind")

    def expand(self):
        raise NotImplementedError

    def set_frozen(self, frozen, recursive=False):
        raise NotImplementedError


class RouterNode(TreeNode):
    def __init__(self, ant, router, left_child, right_child, transformer_count):
        super().__init__(ant, router.in_shape, router.in_shape, transformer_count)
        self.left_child = left_child
        self.right_child = right_child
        self.unexpanded_depth = depth_inc(
            depth_min(
                self.left_child.unexpanded_depth, self.right_child.unexpanded_depth
            )
        )
        self.router = router

    def tree_state_dict(self):
        return {
            "kind": "router",
            "in_shape": self.in_shape,
            "out_shape": self.out_shape,
            "left_child": self.left_child.tree_state_dict(),
            "right_child": self.right_child.tree_state_dict(),
            "transformer_count": self.transformer_count,
        }

    @classmethod
    def load_tree_state_dict(cls, ant, tree_state_dict):
        assert tree_state_dict["kind"] == "router"
        return cls(
            ant,
            ant.new_router(tree_state_dict["in_shape"]),
            TreeNode.load_tree_state_dict(ant, tree_state_dict["left_child"]),
            TreeNode.load_tree_state_dict(ant, tree_state_dict["right_child"]),
            tree_state_dict["transformer_count"],
        )

    def get_tree_composition(self):
        num_router, num_transformer, num_solver = self.left_child.get_tree_composition()
        r = self.right_child.get_tree_composition()
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

        if self.ant.stochastic:
            r = torch.rand(p.size())
        else:
            r = 0.5

        
        left_mask = (p > r).squeeze()
        right_mask = (~(p > r)).squeeze()
        l = self.left_child(x[left_mask])
        r = self.right_child(x[right_mask])

        o = l.new_empty(p.size()[:1] + l.size()[1:])
        if l.size()[0]: o[left_mask] = l
        if r.size()[0]: o[right_mask] = r
        return o

    def set_frozen(self, frozen, recursive=False):
        for param in self.router.parameters():
            param.requires_grad = not frozen

        if recursive:
            self.left_child.set_frozen(frozen, True)
            self.right_child.set_frozen(frozen, True)


class TransformerNode(TreeNode):
    def __init__(self, ant, transformer, child, transformer_count):
        super().__init__(
            ant, transformer.in_shape, transformer.out_shape, transformer_count
        )
        self.child = child
        self.unexpanded_depth = depth_inc(self.child.unexpanded_depth)
        self.transformer = transformer

    def tree_state_dict(self):
        return {
            "kind": "transformer",
            "in_shape": self.in_shape,
            "out_shape": self.out_shape,
            "child": self.child.tree_state_dict(),
            "transformer_count": self.transformer_count,
        }

    @classmethod
    def load_tree_state_dict(cls, ant, tree_state_dict):
        assert tree_state_dict["kind"] == "transformer"
        return cls(
            ant,
            ant.new_transformer(tree_state_dict["in_shape"], tree_state_dict["transformer_count"]),
            TreeNode.load_tree_state_dict(ant, tree_state_dict["child"]),
            tree_state_dict["transformer_count"],
        )

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
    def __init__(self, ant, solver, transformer_count, depth=0):
        super().__init__(ant, solver.in_shape, (1,), transformer_count)
        self.unexpanded_depth = 0
        self.solver = solver
        self.depth = depth

    def tree_state_dict(self):
        return {
            "kind": "solver",
            "in_shape": self.in_shape,
            "out_shape": self.out_shape,
            "transformer_count": self.transformer_count,
        }

    @classmethod
    def load_tree_state_dict(cls, ant, tree_state_dict):
        assert tree_state_dict["kind"] == "solver"
        return cls(
            ant,
            ant.new_solver(
                tree_state_dict["in_shape"],
                ant.num_classes,
            ),
            tree_state_dict["transformer_count"],
        )

    def get_tree_composition(self):
        return (0, 0, 1)

    def expand(self, train_loader, val_loader, max_expand_epochs, *, device,
            verbose, hist):
        if verbose:
            print("Attempting to expand leaf node...")

        # Mark leaf as expanded.
        self.unexpanded_depth = None

        if self.depth >= self.ant.max_depth:
            return self

        # Use pre-trained existing leaf (but don't re-train it).
        self.set_frozen(True)
        leaf_candidate = self.solver

        # Create transformer.
        t = self.ant.new_transformer(self.in_shape, self.transformer_count)
        s = SolverNode(
            self.ant,
            self.ant.new_solver(t.out_shape, self.ant.num_classes),
            self.transformer_count + 1,
            self.depth + 1
        )
        if self.ant.transformer_inherit:
            s.solver.load_state_dict(leaf_candidate.state_dict())
        transformer_candidate = TransformerNode(self.ant, t, s, self.transformer_count)

        # Create router.
        r = self.ant.new_router(self.in_shape)
        s1 = SolverNode(
            self.ant,
            self.ant.new_solver(r.out_shape, self.ant.num_classes),
            self.transformer_count,
            self.depth + 1
        )
        s2 = SolverNode(
            self.ant,
            self.ant.new_solver(r.out_shape, self.ant.num_classes),
            self.transformer_count,
            self.depth + 1
        )
        if self.ant.router_inherit:
            s1.solver.load_state_dict(leaf_candidate.state_dict())
            s2.solver.load_state_dict(leaf_candidate.state_dict())
        router_candidate = RouterNode(self.ant, r, s1, s2, self.transformer_count)

        multi_head_loss = lambda outputs, labels: torch.stack(
            [self.ant.loss_function(output_head, labels) for output_head in outputs]
        )

        try:
            if verbose:
                print("done creating the new modules")
            # Monkey-patch this nodes' solver.
            self.solver = Stack(leaf_candidate, transformer_candidate, router_candidate)

            # And train.
            optimizer = self.ant.new_optimizer(self.ant.root.parameters())
            batch_scheduler = self.ant.new_batch_scheduler(optimizer)
            epoch_scheduler = self.ant.new_epoch_scheduler(optimizer)
            hist_val_losses = []
            def hist_epoch_scheduler(val_loss):
                hist_val_losses.append(val_loss)
                hist["growth_endtimes"].append(time.time() - hist["training_start"])
                if epoch_scheduler is not None:
                    epoch_scheduler(val_loss)
            ops.train(
                self.ant.root,
                train_loader,
                multi_head_loss,
                optimizer,
                max_expand_epochs,
                batch_scheduler=batch_scheduler,
                epoch_scheduler=hist_epoch_scheduler if hist is not None else epoch_scheduler,
                val_loader=val_loader,
                device=device,
                verbose=verbose,
                patience=self.ant.growth_patience,
            )

            val_losses = ops.eval(
                self.ant.root, val_loader, multi_head_loss, device=device
            )
        finally:
            # Restore self always.
            self.solver = leaf_candidate

        # Use best node.
        best = val_losses.argmin().item()
        if verbose:
            print(
                "Best choice was {} with respective losses {} for leaf/transformer/router.".format(
                    ["leaf", "transformer", "router"][best],
                    "/".join(f"{loss:.5}" for loss in val_losses.tolist()),
                )
            )
        if best != 0 and val_losses[best].item() > val_losses[0].item() * 0.99:
            print("However, no meaningful improvement compared to leaf, pruning.")
            best = 0

        if not self.ant.use_router and best == 2:
            print("However the mode dont use router was enabled so choosing next best option")
            best = val_losses[:-1].argmin().item() # select the best (non-router) module
            print("Best choice is now {}".format(["leaf", "transformer", "router"][best]))

        hist["growth_val_losses"].extend(l.tolist()[best] for l in
            hist_val_losses)

        if val_losses[best].item() < self.ant.best_val_loss:
            pickle.dump(self.ant.state_dict(),  open(f"{self.ant.ant_name}-state-dict.p", "wb"))
            self.ant.best_val_loss = val_losses[best].item()
            if verbose:
                print('Storing new best ANT')

        return [self, transformer_candidate, router_candidate][best]

    def forward(self, x):
        return self.solver(x)

    def set_frozen(self, frozen, recursive=False):
        for param in self.solver.parameters():
            param.requires_grad = not frozen
