import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import argparse
import os
import numpy as np
from pprint import pprint
from utils import set_seed, make_and_restore_model, infer_exp_name
from train import poison_train_model, eval_model
from unlearnable import get_poisoned_loader


def make_data_clean(args):
    if args.dataset == "CIFAR10":
        train_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transforms.ToTensor())
    elif args.dataset == "CIFAR100":
        train_set = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transforms.ToTensor())
    elif args.dataset == "SVHN":
        train_set = datasets.SVHN(args.data_path, split="train", transform=transforms.ToTensor())
        test_set = datasets.SVHN(args.data_path, split="test", transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def make_data_poison(args):
    if args.dataset == "CIFAR10":
        if args.poison_type == "em":
            train_loader, test_loader = get_poisoned_loader(dataset='CIFAR10', batch_size=args.batch_size,
                                                            root=args.data_path,
                                                            noise_path="./exp_data/cifar10/em8/em-fin-def-noise.pkl")
        if args.poison_type == "rem8-2":
            train_loader, test_loader = get_poisoned_loader(dataset='CIFAR10', batch_size=args.batch_size,
                                                            root=args.data_path,
                                                            noise_path="./exp_data/cifar10/rem8-2/rem-fin-def-noise.pkl")
        if args.poison_type == "rem8-4":
            train_loader, test_loader = get_poisoned_loader(dataset='CIFAR10', batch_size=args.batch_size,
                                                            root=args.data_path,
                                                            noise_path="./exp_data/cifar10/rem8-4/rem-fin-def-noise.pkl")
    if args.dataset == "CIFAR100":
        if args.poison_type == "em":
            train_loader, test_loader = get_poisoned_loader(dataset='CIFAR100', batch_size=args.batch_size,
                                                            root=args.data_path,
                                                            noise_path="./exp_data/cifar100/em8/em-fin-def-noise.pkl")
        if args.poison_type == "rem8-2":
            train_loader, test_loader = get_poisoned_loader(dataset='CIFAR100', batch_size=args.batch_size,
                                                            root=args.data_path,
                                                            noise_path="./exp_data/cifar100/rem8-2/rem-fin-def-noise.pkl")
        if args.poison_type == "rem8-4":
            train_loader, test_loader = get_poisoned_loader(dataset='CIFAR100', batch_size=args.batch_size,
                                                            root=args.data_path,
                                                            noise_path="./exp_data/cifar100/rem8-4/rem-fin-def-noise.pkl")
    if args.dataset == "SVHN":
        if args.poison_type == "em":
            train_loader, test_loader = get_poisoned_loader(dataset='SVHN', batch_size=args.batch_size,
                                                            root=args.data_path,
                                                            noise_path="./exp_data/svhn/em8/em-fin-def-noise.pkl")
        if args.poison_type == "rem8-2":
            train_loader, test_loader = get_poisoned_loader(dataset='SVHN', batch_size=args.batch_size,
                                                            root=args.data_path,
                                                            noise_path="./exp_data/svhn/rem8-2/rem-fin-def-noise.pkl")
        if args.poison_type == "rem8-4":
            train_loader, test_loader = get_poisoned_loader(dataset='SVHN', batch_size=args.batch_size,
                                                            root=args.data_path,
                                                            noise_path="./exp_data/svhn/rem8-4/rem-fin-def-noise.pkl")
    return train_loader, test_loader


def main(args):
    clean_train_loader, clean_test_loader = make_data_clean(args)
    poison_train_loader, poison_test_loader = make_data_poison(args)
    set_seed(args.seed)
    if not os.path.isfile(args.model_path):
        model = make_and_restore_model(args.arch, args.dataset)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        writer = SummaryWriter(args.tensorboard_path)
        poison_train_model(args, model, optimizer, poison_train_loader, clean_test_loader, clean_train_loader,
                           poison_test_loader, writer)
    else:
        model, resume_epoch = make_and_restore_model(args.arch, args.dataset, resume_path=args.model_path)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        writer = SummaryWriter(args.tensorboard_path)
        poison_train_model(args, model, optimizer, poison_train_loader, clean_test_loader, clean_train_loader,
                           poison_test_loader, writer, resume_epoch)
    print("clean test loader******************")
    eval_model(args, model, clean_test_loader)
    print("clean train loader******************")
    eval_model(args, model, clean_train_loader)

    print("poison train loader******************")
    eval_model(args, model, poison_train_loader)
    print("poison test loader**************")
    eval_model(args, model, poison_test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('AT Training classifiers on rem of em poisoned dataset')
    parser.add_argument('--poison_type', default='em', choices=['em', 'rem8-2', 'rem8-4'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out_dir', default='./trash', type=str)
    parser.add_argument('--data_path', default='./datasets', type=str)
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', "SVHN"])
    parser.add_argument('--train_loss', default='AT', type=str, choices=['ST', 'AT'])
    parser.add_argument('--eps', default=4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--arch', default='ResNet18', type=str,
                        choices=['VGG16', 'ResNet18', 'WRN28-10'])
    parser.add_argument('--constraint', default='Linf', type=str, choices=['Linf', 'L2'])

    args = parser.parse_args()
    # Training options
    args.batch_size = 128
    args.lr = 0.01
    args.weight_decay = 5e-4
    args.log_gap = 1

    # Attack options
    args.eps = args.eps / 255
    args.step_size = args.eps / 4
    args.num_steps = 10
    args.random_restarts = 1

    # Miscellaneous
    args.poison_name = args.poison_type
    args.out_dir = os.path.join(args.out_dir, args.dataset)
    args.data_path = os.path.join(args.data_path, args.dataset)
    args.exp_name = infer_exp_name(args.train_loss, args.eps, args.epochs, args.arch, args.poison_type, args.seed)
    args.tensorboard_path = os.path.join(args.out_dir, args.exp_name, 'tensorboard')
    args.model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pth')

    pprint(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main(args)


