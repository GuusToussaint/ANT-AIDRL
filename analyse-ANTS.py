import sys
import os
import pickle
from Presets import Presets
from matplotlib import pyplot as plt

ANT_types = [
    "ANT-MNIST-A", "ANT-MNIST-A-CNN", "ANT-MNIST-A-HME",
    "ANT-MNIST-B", "ANT-MNIST-B-CNN", "ANT-MNIST-B-HME", 
    "ANT-MNIST-C", "ANT-MNIST-C-CNN", "ANT-MNIST-C-HME",
    "ANT-CIFAR10-A", "ANT-CIFAR10-A-CNN", "ANT-CIFAR10-A-HME",
    "ANT-CIFAR10-B", "ANT-CIFAR10-B-CNN", "ANT-CIFAR10-B-HME",
    "ANT-CIFAR10-C", "ANT-CIFAR10-C-CNN", "ANT-CIFAR10-C-HME",
    "ANT-SARCOS", "ANT-SARCOS-CNN", "ANT-SARCOS-HME"
    ]

def get_performance(ant_type, root):
    ANT = Presets(ant_type)
    ANT.load_tree(root)
    
    single_path_acc, multi_path_acc = ANT.get_accuracy()
    performance_type = "accuracy"

    if "SARCOS" not in ant_type:
        single_path_acc = (1 - single_path_acc) * 100 # get the error
        multi_path_acc = (1 - multi_path_acc) * 100 # get the error

    print(f"single path {performance_type}:\t{single_path_acc:.2f}\n"
          f"multi path {performance_type}: \t{multi_path_acc:.2f}")


def show_training(ant_type, root):
    training_data = pickle.load(open(os.path.join(root, f'{ant_type}-hist.p'), "rb"))
    print(training_data.keys())

    total_lenth = len(training_data['growth_val_losses']) + len(training_data['refinement_val_losses'])
    plt.plot(
        range(total_lenth),
        training_data['growth_val_losses'] + training_data['refinement_val_losses']
        )
    plt.axvline(len(training_data['growth_val_losses']), linestyle='--')
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.savefig(f'{ant_type}-loss.pdf')

if __name__ == "__main__":

    # read the args
    num_of_args = len(sys.argv)
    if num_of_args != 2:
        raise RuntimeError('incorrect number of arguments passed to function\n'
                           f'Usage: python ANT-experiments.py <one of {ANT_types}>')

    ant_type = sys.argv[1]
    if ant_type not in ANT_types:
        raise RuntimeError('Wrong preset passed to function\n'
                           f"{ant_type} not in {ANT_types}")

    pre_trained_ants = os.listdir('trained-ANTS')
    if f"{ant_type}-hist.p" not in pre_trained_ants:
        raise RuntimeError(f'no pre-trained hist file found for {ant_type}')
    hist_file = pickle.load(open(os.path.join('trained-ANTS', f"{ant_type}-hist.p"), "rb"))

    if f"{ant_type}-state-dict.p" not in pre_trained_ants:
        raise RuntimeError(f'no pre-trained state-dict file found for {ant_type}')


    print(f"Done loading the pre-trained files\nAnalysing results for {ant_type}")
    show_training(ant_type, 'trained-ANTS')
    # show_training(ant_type, '')
    get_performance(ant_type, 'trained-ANTS')