import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import argparse
import os
import numpy as np
from pprint import pprint

from utils import set_seed, make_and_restore_model
from utils import infer_poison_name, infer_exp_name
from train import train_model, eval_model


def make_data(args):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    transform_test = transforms.ToTensor()
    if args.dataset == "CIFAR10":
        train_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        r_test_set = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_train)
        whole_set = train_set.__add__(r_test_set)
        test_set = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_test)
    if args.dataset == "CIFAR100":
        train_set = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        r_test_set = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_train)
        whole_set = train_set.__add__(r_test_set)
        test_set = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(whole_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def main(args):
    train_loader, test_loader = make_data(args)
    set_seed(args.seed)
    if not os.path.isfile(args.model_path):
        model = make_and_restore_model(args.arch, args.dataset)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_step)
        writer = SummaryWriter(args.tensorboard_path)
        train_model(args, model, optimizer, train_loader, test_loader, writer, schedule=schedule)

    model, _ = make_and_restore_model(args.arch, args.dataset, resume_path=args.model_path)
    eval_model(args, model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training classifiers for crafting poison')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out_dir', default='./test', type=str)
    parser.add_argument('--dataset', type=str, default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100"],
                        help='choose the dataset')
    parser.add_argument('--train_loss', default='AT', type=str, choices=['ST', 'AT'])
    parser.add_argument('--eps', default=2, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--arch', default='ResNet18', type=str,
                        choices=['VGG16', 'ResNet18', 'WRN28-10'])
    parser.add_argument('--constraint', default='Linf', type=str, choices=['Linf', 'L2'])
    args = parser.parse_args()

    # Training options
    args.batch_size = 128
    args.lr = 0.1
    args.lr_step = 0.1
    args.lr_milestones = [100, 125]
    args.weight_decay = 5e-4
    args.log_gap = 1

    # Attack options
    args.eps = args.eps / 255
    args.step_size = args.eps / 4
    args.num_steps = 10
    args.random_restarts = 1

    # Miscellaneous
    args.data_path = os.path.join('../datasets', args.dataset)
    args.out_dir = os.path.join(args.out_dir, args.dataset)
    args.exp_name = infer_exp_name(args.train_loss, args.eps, args.epochs, args.arch, 'Clean', args.seed)
    args.tensorboard_path = os.path.join(args.out_dir, args.exp_name, 'tensorboard')
    args.model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pth')

    pprint(vars(args))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
    main(args)
