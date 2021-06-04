import sys
from ant import ANT
from ant.routers import FullyConnectedSigmoidRouter, Conv2DGAPFCSigmoidRouter
from ant.transformers import (
    IdentityTransformer,
    FullyConnectedTransformer,
    Conv2DRelu,
)
from ant.solvers import LinearClassifier, LinearRegressor
import functools
import torchvision
import torch
from torch.utils.data import random_split
import pickle
from sklearn.datasets import fetch_openml
import numpy as np
from loadSARCOS import SARCOSDataset

ANT_types = ["ANT-MNIST-A", "ANT-MNIST-B", "ANT-MNIST-B",
             "ANT-CIFAR10-A", "ANT-CIFAR10-B", "ANT-CIFAR10-C",
             "ANT-SARCOS"]


def setup_data(dataset):
    if dataset == "MNIST":
        num_classes = 10
        max_final_epochs = 100

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5), (0.5)),
            ]
        )

        trainset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )
        testset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
        
        sample_size = len(trainset)
        # sample_size = 500
        trainset, _ = random_split(trainset, [sample_size, len(trainset) - sample_size])

        train_size = int(len(trainset) * 0.9)
        trainset, valset = random_split(trainset, [train_size, len(trainset) - train_size])
        return trainset, testset, valset, num_classes, max_final_epochs
    elif dataset == "CIFAR10":
        num_classes = 10
        max_final_epochs = 200

        # Not really sure that this is how they do it in the original work
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )

        sample_size = len(trainset)
        # sample_size = 500
        trainset, _ = random_split(trainset, [sample_size, len(trainset) - sample_size])

        train_size = int(len(trainset) * 0.9)
        trainset, valset = random_split(trainset, [train_size, len(trainset) - train_size])
        return trainset, testset, valset, num_classes, max_final_epochs
    elif dataset == "SARCOS":
        num_classes = 7
        max_final_epochs = 300

        # Loading the SARCOS dataset from .mat file (since no dataloader is present for this dataset)
        trainset = SARCOSDataset(
            root='./data', 
            train=True
        )

        testset = SARCOSDataset(
            root='./data',
            train=False
        )

        train_size = int(len(trainset) * 0.9)
        trainset, valset = random_split(trainset, [train_size, len(trainset) - train_size])

        return trainset, testset, valset, num_classes, max_final_epochs


class Presets():
    def __init__(self, ant_type):
        self.type = ant_type
        self.datasetname = ant_type.split('-')[1]
        self.trainset, self.testset, self.valset, self.num_classes, self.max_final_epochs = setup_data(self.datasetname)
        print(self.testset[0][0].shape)
        self.tree = self.get_tree()
        self.hist = None

    def start_training(self):
        self.hist = self.tree.fit(
            self.trainset,
            self.valset,
            max_expand_epochs=100,
            max_final_epochs=self.max_final_epochs,
            output_graphviz=f"{self.type}.gv"
        )

    def get_accuracy(self):
        multi_path_acc = self.tree.eval_acc(self.testset)

        self.tree.soft_decision = False
        single_path_acc = self.tree.eval_acc(self.testset)
        return single_path_acc, multi_path_acc

    def save_tree(self):
        state_dict = self.tree.state_dict()
        if self.hist is not None:
            pickle.dump(self.hist, open(f"{self.type}-hist.p", "wb"))
        pickle.dump(state_dict, open(f"{self.type}-state-dict.p", "wb"))

    def load_tree(self, fname):
        state_dict = pickle.load(open(f"{self.type}.p", "rb"))
        self.tree.load_state_dict(state_dict)

    def get_tree(self):
        if self.type == "ANT-MNIST-A":
            # Router:           1 × conv5-40 + GAP + 2×FC +Sigmoid
            # Transformer:      1 × conv5-40 + ReLU
            # Downsample Freq:  1
            return functools.partial(
                        ANT,
                        in_shape=self.trainset[0][0].shape,
                        num_classes=self.num_classes,
                        new_router=functools.partial(
                            Conv2DGAPFCSigmoidRouter,
                            convolutions=1,
                            kernels=40,
                            kernel_size=5,
                            fc_layers=2,
                        ),
                        new_transformer=functools.partial(
                            Conv2DRelu, convolutions=1, kernels=40, kernel_size=5, down_sample_freq=1
                        ),
                        new_solver=LinearClassifier,
                        new_optimizer=lambda in_shape: torch.optim.Adam(in_shape, lr=1e-3, betas=(0.9, 0.999)),
                    )()
        elif self.type == "ANT-MNIST-B":
            # Router:           1 × conv3-40 + GAP + 2×FC +Sigmoid
            # Transformer:      1 × conv3-40 +ReLU
            # Downsample Freq:  2
            return functools.partial(
                        ANT,
                        in_shape=self.trainset[0][0].shape,
                        num_classes=self.num_classes,
                        new_router=functools.partial(
                            Conv2DGAPFCSigmoidRouter,
                            convolutions=1,
                            kernels=40,
                            kernel_size=3,
                            fc_layers=2,
                        ),
                        new_transformer=functools.partial(
                            Conv2DRelu, convolutions=1, kernels=40, kernel_size=3, down_sample_freq=2
                        ),
                        new_solver=LinearClassifier,
                        new_optimizer=lambda in_shape: torch.optim.Adam(in_shape, lr=1e-3, betas=(0.9, 0.999)),
                    )()
        elif self.type == "ANT-MNIST-C":
            # Router:           1 × conv5-5 + GAP + 2×FC +Sigmoid
            # Transformer:      1 × conv5-5+ReLU
            # Downsample Freq:  2
            return functools.partial(
                        ANT,
                        in_shape=self.trainset[0][0].shape,
                        num_classes=self.num_classes,
                        new_router=functools.partial(
                            Conv2DGAPFCSigmoidRouter,
                            convolutions=1,
                            kernels=5,
                            kernel_size=5,
                            fc_layers=2,
                        ),
                        new_transformer=functools.partial(
                            Conv2DRelu, convolutions=1, kernels=5, kernel_size=5, down_sample_freq=2
                        ),
                        new_solver=LinearClassifier,
                        new_optimizer=lambda in_shape: torch.optim.Adam(in_shape, lr=1e-3, betas=(0.9, 0.999)),
                    )()
        elif self.type == "ANT-CIFAR10-A":
            # Router:           2 × conv3-128 + GAP + 1×FC + Sigmoid
            # Transformer:      2 × conv3-128 + ReLU
            # Solver:           GAP + LC
            # Downsample Freq:  1
            return functools.partial(
                        ANT,
                        in_shape=self.trainset[0][0].shape,
                        num_classes=self.num_classes,
                        new_router=functools.partial(
                            Conv2DGAPFCSigmoidRouter,
                            convolutions=2,
                            kernels=128,
                            kernel_size=3,
                            fc_layers=1,
                        ),
                        new_transformer=functools.partial(
                            Conv2DRelu, convolutions=2, kernels=128, kernel_size=3, down_sample_freq=1
                        ),
                        new_solver=functools.partial(LinearClassifier, GAP=True),
                        new_optimizer=lambda in_shape: torch.optim.Adam(in_shape, lr=1e-3, betas=(0.9, 0.999)),
                    )()
        elif self.type == "ANT-CIFAR10-B":
            # Router:           2 × conv3-96 + GAP + 1×FC + Sigmoid
            # Transformer:      2 × conv3-96 + ReLU
            # Solver:           LC
            # Downsample Freq:  1
            return functools.partial(
                        ANT,
                        in_shape=self.trainset[0][0].shape,
                        num_classes=self.num_classes,
                        new_router=functools.partial(
                            Conv2DGAPFCSigmoidRouter,
                            convolutions=2,
                            kernels=96,
                            kernel_size=3,
                            fc_layers=1,
                        ),
                        new_transformer=functools.partial(
                            Conv2DRelu, convolutions=2, kernels=96, kernel_size=3, down_sample_freq=1
                        ),
                        new_solver=functools.partial(LinearClassifier, GAP=False),
                        new_optimizer=lambda in_shape: torch.optim.Adam(in_shape, lr=1e-3, betas=(0.9, 0.999)),
                    )()
        elif self.type == "ANT-CIFAR10-C":
            # Router:           2 × conv3-48 + GAP + 1×FC + Sigmoid
            # Transformer:      2 × conv3-96 + ReLU
            # Solver:           GAP + LC
            # Downsample Freq:  1
            return functools.partial(
                        ANT,
                        in_shape=self.trainset[0][0].shape,
                        num_classes=self.num_classes,
                        new_router=functools.partial(
                            Conv2DGAPFCSigmoidRouter,
                            convolutions=2,
                            kernels=48,
                            kernel_size=3,
                            fc_layers=1,
                        ),
                        new_transformer=functools.partial(
                            Conv2DRelu, convolutions=2, kernels=96, kernel_size=3, down_sample_freq=1
                        ),
                        new_solver=functools.partial(LinearClassifier, GAP=True),
                        new_optimizer=lambda in_shape: torch.optim.Adam(in_shape, lr=1e-3, betas=(0.9, 0.999)),
                    )()
        elif self.type == "ANT-SARCOS":
            # Router:           1 × FC + Sigmoid
            # Transformer:      1 x FC + tahn
            # Solver:           LR
            # Downsample Freq:  0
            return functools.partial(
                ANT,
                in_shape=self.trainset[0][0].shape,
                num_classes=self.num_classes,
                new_router=functools.partial(
                    FullyConnectedSigmoidRouter,
                ),
                new_transformer=functools.partial(
                    FullyConnectedTransformer,
                ),
                new_solver=LinearRegressor,
                new_optimizer=lambda in_shape: torch.optim.Adam(in_shape, lr=1e-3, betas=(0.9, 0.999)),
                regression=True
            )()
        raise RuntimeError('The defined preset is not available ')


def get_ant_type():
    # checking arguments
    num_of_args = len(sys.argv)
    if num_of_args != 2:
        raise RuntimeError('incorrect number of arguments passed to function\n'
                           f'Usage: python ANT-experiments.py <one of {ANT_types}>')

    ant_type = sys.argv[1]
    if ant_type not in ANT_types:
        raise RuntimeError('Wrong preset passed to function\n'
                           f"{ant_type} not in {ANT_types}")

    return ant_type


if __name__ == '__main__':
    ant_type = get_ant_type()

    ANT = Presets(ant_type)
    ANT.start_training()
    # ANT.load_tree(f"{ant_type}.p")
    
    ANT.save_tree()
    single_acc, multi_acc = ANT.get_accuracy()
    print(f"single path acc:\t{single_acc}\n"
          f"multi path acc: \t{multi_acc}")