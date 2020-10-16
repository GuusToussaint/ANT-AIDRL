from ant import ANT
from ant.routers import FullyConnectedSigmoidRouter, Conv2DFCSigmoid
from ant.transformers import (
    IdentityTransformer,
    FullyConnected1DTransformer,
    Conv2DRelu,
)
from ant.solvers import FullyConnectedSolver, Linear2DSolver

import functools
import torchvision
import torch
from ant import ops
import numpy as np


import random

if __name__ == "__main__":
    random.seed(421)

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     torchvision.transforms.Lambda(torch.flatten)])

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    num_classes = 10

    # transform = torchvision.transforms.Compose(
    #     [
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]
    # )
    # trainset = torchvision.datasets.CIFAR10(
    #     root="./data", train=True, download=True, transform=transform
    # )
    # testset = torchvision.datasets.CIFAR10(
    #     root="./data", train=False, download=True, transform=transform
    # )

    # For quick debugging.
    # trainset = torch.utils.data.Subset(trainset, range(1024))
    # testset = torch.utils.data.Subset(testset, range(1024))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=16, shuffle=True, num_workers=1
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=1
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    t = ANT(
        in_shape=trainset[0][0].shape,
        num_classes=len(trainset.classes),
        new_router=functools.partial(
            Conv2DFCSigmoid, convolutions=1, kernels=40, kernel_size=5
        ),
        # new_transformer=functools.partial(
        #     Conv2DRelu, convolutions=1, kernels=40, kernel_size=5
        # ),
        new_transformer=IdentityTransformer,
        new_solver=FullyConnectedSolver,
        # new_optimizer=torch.optim.Adam,
        new_optimizer=lambda in_shape: torch.optim.SGD(in_shape, lr=0.01),
    )

    t.do_train(
        trainloader, testloader, max_expand_epochs=5, max_final_epochs=5, device=device
    )
    total = 0
    correct = 0
    with torch.no_grad():
        t.root.eval()
        t.root.to(device)
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = t.root(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of the network : %d %%" % (100 * correct / total))
