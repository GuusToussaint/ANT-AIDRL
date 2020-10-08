from ant import ANT
from ant.routers import FullyConnectedSigmoidRouter, Conv2DFCSigmoid
from ant.transformers import FullyConnected1DTransformer, Conv2DRelu
from ant.solvers import Linear1DSolver, Linear2DSolver

import functools
import torchvision
import torch


import random

if __name__ == "__main__":
    random.seed(420)

    # transform = torchvision.transforms.Compose(
    #     [torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     torchvision.transforms.Lambda(torch.flatten)])
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)


    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    t = ANT(in_shape=trainset[0][0].shape, num_classes=len(classes),
            new_router = functools.partial(Conv2DFCSigmoid, convolutions=1, kernels=40, kernel_size=5),
            new_transformer = functools.partial(Conv2DRelu, convolutions=1, kernels=40, kernel_size=5),
            new_solver = Linear2DSolver,
            new_optimizer = torch.optim.Adam)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t.do_train(trainloader, testloader, expand_epochs=3, final_epochs=5, device=device)


    total = 0
    correct = 0
    with torch.no_grad():
        t.root.eval()
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = t.root(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network : %d %%' % (100 * correct / total)) 
