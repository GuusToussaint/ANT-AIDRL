import torch
import torch.nn as nn
from graphviz import Digraph
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import Tensor
import gc
import numpy as np
import random

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
            self.loss_function = lambda pred, target: nll_loss(pred, target)

    def do_train(
        self,
        train_loader,
        val_loader,
        max_expand_epochs,
        max_final_epochs,
        *,
        device="cpu",
        verbose=True,
        output_graphviz=None,
    ):
        self.training = True

        try:
            if verbose:
                print("Starting tree build process.")

            # Create initial leaf and train it.
            self.root = SolverNode(
                self, self.new_solver(self.in_shape, self.num_classes), 0
            )
            ops.train(
                self.root,
                train_loader,
                self.loss_function,
                self.new_optimizer,
                max_expand_epochs,
                val_loader=val_loader,
                device=device,
                verbose=verbose,
                patience=5,
            )

            # Expand while possible.
            while not self.root.fully_expanded():
                try:
                    old = self.state_dict()

                    self.root = self.root.expand(
                        train_loader,
                        val_loader,
                        max_expand_epochs,
                        device=device,
                        verbose=verbose,
                    )

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

            # Final refinement.
            if verbose:
                print("Starting final refinement.")

            self.root.set_frozen(False, recursive=True)
            ops.train(
                self.root,
                train_loader,
                self.loss_function,
                self.new_optimizer,
                max_final_epochs,
                val_loader=val_loader,
                device=device,
                verbose=verbose,
                lr_factor=0.1,
            )

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
        dataset,
        batch_size=512,
        verbose=True,
        max_expand_epochs=100,
        max_final_epochs=200,
        transform=None,
        val_size=0.1,
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        val_instances = int(len(dataset) * val_size)
        trainset, valset = random_split(
            dataset,
            [len(dataset) - val_instances, val_instances],
            generator=torch.Generator(),
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        if verbose:
            print(
                f"Fitting model with {len(trainset)} training instances and {len(valset)} validation instances."
            )

        self.do_train(
            trainloader,
            valloader,
            max_expand_epochs=max_expand_epochs,
            max_final_epochs=max_final_epochs,
            device=device,
        )

    def eval(self, dataset, batch_size=16):
        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        total = 0
        correct = 0
        with torch.no_grad():
            self.root.eval()
            self.root.to(device)
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = self.root(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Accuracy of the network : %f %%" % (100 * correct / total))
        print(f"total {total}, correct {correct}, wrong {total-correct}")


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
        }

    @classmethod
    def load_tree_state_dict(cls, ant, tree_state_dict):
        assert tree_state_dict["kind"] == "router"
        return cls(
            ant,
            ant.new_router(tree_state_dict["in_shape"]),
            TreeNode.load_tree_state_dict(ant, tree_state_dict["left_child"]),
            TreeNode.load_tree_state_dict(ant, tree_state_dict["right_child"]),
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
            r = torch.rand(x.size())
        else:
            r = 0.5

        o = x.new_empty(x.size())
        left_mask = p < r
        right_mask = ~(p < r)
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
            ant.new_transformer(tree_state_dict["in_shape"]),
            TreeNode.load_tree_state_dict(ant, tree_state_dict["child"]),
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
    def __init__(self, ant, solver, transformer_count):
        super().__init__(ant, solver.in_shape, (1,), transformer_count)
        self.unexpanded_depth = 0
        self.solver = solver

    def tree_state_dict(self):
        return {
            "kind": "solver",
            "in_shape": self.in_shape,
            "out_shape": self.out_shape,
        }

    @classmethod
    def load_tree_state_dict(cls, ant, tree_state_dict):
        assert tree_state_dict["kind"] == "solver"
        return cls(ant, ant.new_solver(tree_state_dict["in_shape"], ant.num_classes))

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
        t = self.ant.new_transformer(self.in_shape, self.transformer_count)
        s = SolverNode(
            self.ant,
            self.ant.new_solver(t.out_shape, self.ant.num_classes),
            self.transformer_count + 1,
        )
        transformer_candidate = TransformerNode(self.ant, t, s, self.transformer_count)
        if self.ant.transformer_inherit:
            s.solver.load_state_dict(leaf_candidate.state_dict())

        # Create router.
        r = self.ant.new_router(self.in_shape)
        s1 = SolverNode(
            self.ant,
            self.ant.new_solver(r.out_shape, self.ant.num_classes),
            self.transformer_count,
        )
        s2 = SolverNode(
            self.ant,
            self.ant.new_solver(r.out_shape, self.ant.num_classes),
            self.transformer_count,
        )
        router_candidate = RouterNode(self.ant, r, s1, s2, self.transformer_count)
        if self.ant.router_inherit:
            s1.solver.load_state_dict(leaf_candidate.state_dict())
            s2.solver.load_state_dict(leaf_candidate.state_dict())

        multi_head_loss = lambda outputs, labels: torch.stack(
            [self.ant.loss_function(output_head, labels) for output_head in outputs]
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
                patience=5,
            )
            val_losses = ops.eval(
                self.ant.root, val_loader, multi_head_loss, device=device
            )
        except Exception as e:
            print(f"error message: {e}")
        finally:
            # Restore self.
            self.solver = leaf_candidate

        # Use best node.
        best = val_losses.argmin().item()
        if round(val_losses[val_losses.argmin().item()].item(), 5) == round(
            val_losses[val_losses.argmax().item()].item(), 5
        ):
            # No 'real' difference, thus choosing lead as best
            best = 0

        if verbose:
            print(
                "Best choice was {} with respective losses {} for leaf/transformer/router.".format(
                    ["leaf", "transformer", "router"][best],
                    "/".join(f"{loss:.5}" for loss in val_losses.tolist()),
                )
            )
        return [self, transformer_candidate, router_candidate][best]

    def forward(self, x):
        return self.solver(x)

    def set_frozen(self, frozen, recursive=False):
        for param in self.solver.parameters():
            param.requires_grad = not frozen
