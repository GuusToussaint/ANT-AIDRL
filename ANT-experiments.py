import sys
from Presets import Presets


ANT_types = [
    "ANT-MNIST-A", "ANT-MNIST-A-CNN", "ANT-MNIST-A-HME",
    "ANT-MNIST-B", "ANT-MNIST-B-CNN", "ANT-MNIST-B-HME", 
    "ANT-MNIST-C", "ANT-MNIST-C-CNN", "ANT-MNIST-C-HME",
    "ANT-CIFAR10-A", "ANT-CIFAR10-A-CNN", "ANT-CIFAR10-A-HME",
    "ANT-CIFAR10-B", "ANT-CIFAR10-B-CNN", "ANT-CIFAR10-B-HME",
    "ANT-CIFAR10-C", "ANT-CIFAR10-C-CNN", "ANT-CIFAR10-C-HME",
    "ANT-SARCOS", "ANT-SARCOS-CNN", "ANT-SARCOS-HME"
    ]


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
    
    ANT.save_tree()
    single_acc, multi_acc = ANT.get_accuracy()
    print(f"single path acc:\t{single_acc}\n"
          f"multi path acc: \t{multi_acc}")