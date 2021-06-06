import sys
import os
import pickle
from Presets import Presets

ANT_types = ["ANT-MNIST-A", "ANT-MNIST-B", "ANT-MNIST-C",
             "ANT-CIFAR10-A", "ANT-CIFAR10-B", "ANT-CIFAR10-C",
             "ANT-SARCOS"]

def get_performance(ant_type, root):
    ANT = Presets(ant_type)
    ANT.load_tree(root)
    
    single_path_acc, multi_path_acc = ANT.get_accuracy()
    performance_type = "accuracy"

    if "SARCOS" in ant_type: # dealing with regression
        single_path_acc = single_path_acc / ANT.num_classes # deviding the MSE (sum) by the number of classes
        multi_path_acc = multi_path_acc / ANT.num_classes # deviding the MSE (sum) by the number of classes
        performance_type = "MSE"
    else:
        single_path_acc = (1 - single_path_acc) * 100 # get the error
        multi_path_acc = (1 - multi_path_acc) * 100 # get the error

    print(f"single path {performance_type}:\t{single_path_acc:.2f}\n"
          f"multi path {performance_type}: \t{multi_path_acc:.2f}")

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
    get_performance(ant_type, 'trained-ANTS')