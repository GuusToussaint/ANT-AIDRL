from ant import ANT
from ant.routers import (
    FullyConnectedSigmoidRouter,
    Conv2DGAPFCSigmoidRouter
)
from ant.transformers import (
    FullyConnectedTransformer,
    Conv2DRelu,
)
from ant.solvers import (
    LinearClassifier,
    LinearRegressor
)
from torch.utils.data import random_split
import functools
import torchvision
import torch
import random


# Function for returning the ANT with correct parameters for presets
def presets(preset, in_shape, num_classes):

    router = None
    transformer = None
    solver = None

    # ANT-SARCOS
    if preset == "ANT-SARCOS":
        router = FullyConnectedSigmoidRouter
        transformer = FullyConnectedTransformer
        solver = LinearRegressor

    # ANT-MNIST-A
    if preset == "ANT-MNIST-A":
        router = functools.partial(
            Conv2DGAPFCSigmoidRouter, convolutions=1, kernel_size=5, kernels=40, fc_layers=2
        )
        transformer = functools.partial(
            Conv2DRelu, convolutions=1, kernel_size=5, kernels=40, down_sample_freq=1
        )
        solver = LinearClassifier

    # ANT-MNIST-B
    if preset == "ANT-MNIST-B":
        router = functools.partial(
            Conv2DGAPFCSigmoidRouter, convolutions=1, kernel_size=3, kernels=40, fc_layers=2
        )
        transformer = functools.partial(
            Conv2DRelu, convolutions=1, kernel_size=3, kernels=40, down_sample_freq=2
        )
        solver = LinearClassifier

    # ANT-MNIST-C
    if preset == "ANT-MNIST-C":
        router = functools.partial(
            Conv2DGAPFCSigmoidRouter, convolutions=1, kernel_size=5, kernels=5, fc_layers=2
        )
        transformer = functools.partial(
            Conv2DRelu, convolutions=1, kernel_size=5, kernels=5, down_sample_freq=2
        )
        solver = LinearClassifier

    # ANT-CIFAR10-A
    if preset == "ANT-CIFAR10-A":
        router = functools.partial(
            Conv2DGAPFCSigmoidRouter, convolutions=2, kernel_size=3, kernels=128, fc_layers=1
        )
        transformer = functools.partial(
            Conv2DRelu, convolutions=2, kernel_size=3, kernels=128, down_sample_freq=1
        )
        solver = functools.partial(
            LinearClassifier, GAP=True
        )

    # ANT-CIFAR10-B
    if preset == "ANT-CIFAR10-B":
        router = functools.partial(
            Conv2DGAPFCSigmoidRouter, convolutions=2, kernel_size=3, kernels=96, fc_layers=1
        )
        transformer = functools.partial(
            Conv2DRelu, convolutions=2, kernel_size=3, kernels=96, down_sample_freq=1
        )
        solver = functools.partial(
            LinearClassifier, GAP=False
        )

    # ANT-CIFAR10-C
    if preset == "ANT-CIFAR10-C":
        router = functools.partial(
            Conv2DGAPFCSigmoidRouter, convolutions=2, kernel_size=3, kernels=48, fc_layers=1
        )
        transformer = functools.partial(
            Conv2DRelu, convolutions=2, kernel_size=3, kernels=48, down_sample_freq=1
        )
        solver = functools.partial(
            LinearClassifier, GAP=True
        )

    if router is None and transformer is None and solver is None:
        return None

    print(router)
    print(transformer)

    tree = ANT(
        in_shape=in_shape,
        num_classes=num_classes,
        new_router=router,
        new_transformer=transformer,
        new_solver=solver,
        new_optimizer=lambda in_shape: torch.optim.Adam(in_shape),
    )
    return tree


if __name__ == "__main__":
    # MNIST dataset
    print("Starting experiments for ANT-MNIST-C")

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5), (0.5))])

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    sample_size = 10000
    trainset, _ = random_split(trainset, [sample_size, len(trainset)-sample_size])

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    num_classes = 10
    in_shape = trainset[0][0].shape
    tree = presets("ANT-MNIST-C", in_shape, num_classes)

    tree.fit(trainset, max_expand_epochs=100, max_final_epochs=100)

    tree.eval(testset)