import argparse


def add_shared_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument('--arch', type=str, default='ResNet18',
                        choices=['VGG16', 'ResNet18', 'WRN28-10'],
                        help='choose the model architecture')
    parser.add_argument('--dataset', type=str, default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100", 'tiny-imagenet'],
                        help='choose the dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='set the batch size')
    parser.add_argument('--train-steps', type=int, default=5000,
                        help='set the training steps')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='set the learning rate decay rate')
    parser.add_argument('--lr-decay-freq', type=int, default=2000,
                        help='set the learning rate decay frequency')

    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='select which optimizer to use')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='set the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='set the weight decay rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='set the momentum for SGD')

    parser.add_argument('--pgd-radius', type=float, default=8,
                        help='set the perturbation radius in pgd')
    parser.add_argument('--pgd-steps', type=int, default=10,
                        help='set the number of iteration steps in pgd')
    parser.add_argument('--pgd-step-size', type=float, default=1.6,
                        help='set the step size in pgd')
    parser.add_argument('--pgd-random-start', action='store_true',
                        help='if select, randomly choose starting points each time performing pgd')
    parser.add_argument('--pgd-norm-type', type=str, default='l-infty',
                        choices=['l-infty', 'l2', 'l1'],
                        help='set the type of metric norm in pgd')

    parser.add_argument('--parallel', action='store_true',
                        help='select to use distributed data parallel')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='for distributed data parallel')

    parser.add_argument('--data-dir', type=str, default='../datasets',
                        help='set the path to the exp data')
    parser.add_argument('--save-dir', type=str, default='./exp_data/CIFAR10/em8',
                        help='set which dictionary to save the experiment result')
    parser.add_argument('--save-name', type=str, default='em',
                        help='set the save name of the experiment result')
