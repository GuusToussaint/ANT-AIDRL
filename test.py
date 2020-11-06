from ant import ANT
from ant.routers import FullyConnectedSigmoidRouter, Conv2DGAPFCSigmoidRouter
from ant.transformers import (
    IdentityTransformer,
    FullyConnected1DTransformer,
    Conv2DRelu,
)
from ant.solvers import FullyConnectedSolver
from sklearn.datasets import fetch_openml
import functools
import torchvision
import torch
from torch.utils.data import random_split
import numpy as np
import random

if __name__ == "__main__":
    random.seed(421)



    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))])


    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # hold out 10% for the validation set
    num_classes = 10

    t = ANT(
        in_shape=trainset[0][0].shape,
        num_classes=num_classes,
        new_router=functools.partial(
            Conv2DGAPFCSigmoidRouter, convolutions=1, kernels=40, kernel_size=5, fc_layers=2
        ),
        new_transformer=functools.partial(
            Conv2DRelu, convolutions=1, kernels=40, kernel_size=5
        ),
        new_solver=FullyConnectedSolver,
        new_optimizer=lambda in_shape: torch.optim.Adam(in_shape),
    )

    t.fit(trainset, transform=transform)



    # TODO: change to tree.predict()
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=1
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total = 0
    correct = 0
    with torch.no_grad():
        t.root.eval()
        t.root.to(device)
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = t.root(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # plt.imshow(inputs[0].cpu().reshape(28, 28, 1))
            # plt.title(f"true:{labels[0].cpu()}\tpredicted:{predicted[0].cpu()}")
            # plt.show()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print("Accuracy of the network : %d %%" % (100 * correct / total))
