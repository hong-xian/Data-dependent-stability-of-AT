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
from utils import infer_exp_name, infer_poison_name, PoisonDataset
from train import poison_train_model, eval_model
from minmin import get_poisoned_loader


transform_train = transforms.Compose([
    transforms.RandomCrop(32, 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.ToTensor()


def make_data_clean(args):
    if args.dataset == "CIFAR10":
        train_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_test)
        test_set = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_test)
    elif args.dataset == "CIFAR100":
        train_set = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_test)
        test_set = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def make_data_poison(args):
    train_set = PoisonDataset(args.poison_path, transform=transform_train)
    test_set = PoisonDataset(args.poison_test_path, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
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
    parser.add_argument('--poison_type', default='Hyp', choices=['Adv', 'Random', 'Hyp'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out_dir', default='./test', type=str)
    parser.add_argument('--data_path', default='./datasets', type=str)
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--train_loss', default='AT', type=str, choices=['ST', 'AT'])
    parser.add_argument('--eps', default=4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--arch', default='ResNet18', type=str,
                        choices=['VGG16', 'ResNet18', 'WRN28-10'])
    parser.add_argument('--constraint', default='Linf', type=str, choices=['Linf', 'L2'])

    parser.add_argument('--poison_steps', default=100, type=int)
    parser.add_argument('--poison_eps', default=8, type=float)
    parser.add_argument('--poison_aug', action='store_true',
                        help='if select, generate poison with data augmentation')
    parser.add_argument('--craft_model_loss', default='ST', type=str, choices=['ST', 'AT'])
    parser.add_argument('--craft_model_eps', default=2, type=float)
    parser.add_argument('--craft_model_epoch', default=150, type=int)
    parser.add_argument('--craft_model_arch', default='ResNet18', type=str)
    args = parser.parse_args()
    # Training options
    args.batch_size = 128
    args.lr = 0.01
    args.weight_decay = 5e-4
    args.log_gap = 1

    # Attack options
    args.eps = args.eps / 255
    args.poison_eps = args.poison_eps / 255
    args.craft_model_eps = args.craft_model_eps / 255
    args.step_size = args.eps / 4
    args.num_steps = 10
    args.random_restarts = 1

    # Miscellaneous
    args.out_dir = os.path.join(args.out_dir, args.dataset)
    args.data_path = os.path.join(args.data_path, args.dataset)
    args.poison_name = infer_poison_name(args.poison_type, args.poison_steps, args.craft_model_loss,
                                         args.craft_model_eps, args.craft_model_epoch, args.craft_model_arch,
                                         args.poison_aug, args.poison_eps)
    args.exp_name = infer_exp_name(args.train_loss, args.eps, args.epochs, args.arch, args.poison_name, args.seed)
    args.tensorboard_path = os.path.join(args.out_dir, args.exp_name, 'tensorboard')
    args.model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pth')

    # for random hyp adv find poison data path
    args.craft_model_exp_name = infer_exp_name(args.craft_model_loss, args.craft_model_eps, args.craft_model_epoch,
                                               args.craft_model_arch, 'Clean')
    args.poison_path = os.path.join(args.out_dir, args.craft_model_exp_name, args.poison_name + '.poison')
    args.poison_test_path = os.path.join(args.out_dir, args.craft_model_exp_name, args.poison_name + '.poison_test')
    pprint(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main(args)


