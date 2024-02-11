import argparse

def str2bool(v):
    return v.lower() in ('true')

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Attack settings.
    # parser.add_argument('', default=, type=, choices=[])
    parser.add_argument('--threshold', default = 0.05, type=float)

    # Model settings.
    parser.add_argument('--model', default='VGG16', type=str, choices=['VGG16', 'resnet18'])
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--max_epoch', default = 100, type=int)
    parser.add_argument('--learning_rate', default = 0.1, type=float)

    # Data settings.
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['MNIST','CIFAR10','GTSRB','TinyImageNet'])
    parser.add_argument('--batch_size', default = 512, type=int)

    args = parser.parse_args()
    return args