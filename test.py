from ant import ANT
from ant.routers import FullyConnectedSigmoidRouter, Conv2DGAPFCSigmoidRouter
from ant.transformers import (
    IdentityTransformer,
    FullyConnectedTransformer,
    Conv2DRelu,
)
from ant.solvers import LinearClassifier
from sklearn.datasets import fetch_openml
import functools
import torchvision
import torch
from torch.utils.data import random_split
import numpy as np
import random

if __name__ == "__main__":
    random.seed(421)
    np.random.seed(421)

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

    train_size = int(len(trainset) * 0.9)
    trainset, valset = random_split(trainset, [train_size, len(trainset) - train_size])

    # hold out 10% for the validation set
    num_classes = 10

    tree_builder = functools.partial(
        ANT,
        in_shape=trainset[0][0].shape,
        num_classes=num_classes,
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
    )

    t = tree_builder()
    hist = t.fit(trainset, valset, max_expand_epochs=100, max_final_epochs=50, output_graphviz="live_tree.gv")
    print("Accuracy of the network: {:%}".format(t.eval_acc(testset)))

    print(hist)

    t2 = tree_builder()
    t2.load_state_dict(t.state_dict())
    print("Accuracy of the network loaded from weights: {:%}".format(t2.eval_acc(testset)))

