import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import argparse
import os
import numpy as np
from pprint import pprint

from utils import set_seed, make_and_restore_model
from utils import infer_exp_name_no
from train import train_model, eval_model
from tiny_imagenet import TinyImageNet


def make_data_clean(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.ToTensor()
    if args.dataset == "CIFAR10":
        train_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_test)
    elif args.dataset == "CIFAR100":
        train_set = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_test)
    elif args.dataset == "SVHN":
        transform_train = transforms.ToTensor()
        train_set = datasets.SVHN(args.data_path, split="train", download=True, transform=transform_train)
        test_set = datasets.SVHN(args.data_path, split='test', download=True, transform=transform_test)
    elif args.dataset == "Tiny-Imagenet":
        transform_train = transforms.Compose([
            # transforms.RandomCrop(64, 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_set = TinyImageNet(root='/data/liushuang/datasets/Tiny-Imagenet/tiny-imagenet-200', train=True,
                                 transform=transform_train)
        test_set = TinyImageNet(root='/data/liushuang/datasets/Tiny-Imagenet/tiny-imagenet-200', train=False,
                                transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def main(args):
    train_loader, test_loader = make_data_clean(args)
    set_seed(args.seed)
    if not os.path.isfile(args.model_path):
        model = make_and_restore_model(args.arch, args.dataset)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        writer = SummaryWriter(args.tensorboard_path)
        if args.schedule:
            schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_step)
            train_model(args, model, optimizer, train_loader, test_loader, writer, schedule=schedule)
        else:
            train_model(args, model, optimizer, train_loader, test_loader, writer)
    else:
        model, resume_epoch = make_and_restore_model(args.arch, args.dataset, resume_path=args.model_path)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        writer = SummaryWriter(args.tensorboard_path)
        if args.schedule:
            schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_step)
            train_model(args, model, optimizer, train_loader, test_loader, writer, resume_epoch, schedule=schedule)
        else:
            train_model(args, model, optimizer, train_loader, test_loader, writer, resume_epoch)
    print("test loader******************")
    eval_model(args, model, test_loader)
    print("train loader******************")
    eval_model(args, model, train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training classifiers for datasets')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'CIFAR100', 'SVHN', 'Tiny-Imagenet'],
                        help='choose the dataset')
    parser.add_argument('--train_loss', default='AT', type=str, choices=['ST', 'AT'])
    parser.add_argument('--eps', default=4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--arch', default='ResNet18', type=str,
                        choices=['VGG16', 'ResNet18', 'WRN28-10'])
    parser.add_argument('--constraint', default='Linf', type=str, choices=['Linf', 'L2'])
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--gpuid', default=5, type=int)
    parser.add_argument('--schedule', action='store_true',
                        help='if select, use lr decay with step 0.1 at [100, 150]')

    args = parser.parse_args()
    args.batch_size = 128
    args.weight_decay = 5e-4
    args.log_gap = 1
    if args.schedule:
        args.lr_step = 0.1
        args.lr_milestones = [100, 150]

    # Attack options
    args.eps = args.eps / 255
    args.step_size = args.eps / 4
    args.num_steps = 10
    args.random_restarts = 1

    args.data_path = os.path.join('../datasets', args.dataset)
    args.out_dir = os.path.join('../test_results', args.dataset)
    args.exp_name = infer_exp_name_no(args.train_loss, args.eps, args.epochs, args.arch, args.seed, args.schedule)
    args.tensorboard_path = os.path.join(args.out_dir, args.exp_name, 'tensorboard')
    args.model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pth')
    args.model_path_last = os.path.join(args.out_dir, args.exp_name, 'checkpoint_last.pth')

    pprint(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main(args)


