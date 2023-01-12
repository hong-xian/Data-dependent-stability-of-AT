import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
import torchvision
from tqdm import tqdm
from pprint import pprint
from utils import set_seed, PoisonDataset, make_and_restore_model, AverageMeter, accuracy_top1
from utils import infer_poison_name, infer_exp_name
from utils import RandomTransform
from attacks.step import LinfStep, L2Step

STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}
params = dict(source_size=32, target_size=32, shift=8, fliplr=True)
trans = RandomTransform(**params, mode='bilinear')


def batch_poison(model, x, target, args):
    orig_x = x.clone().detach()
    step = STEPS[args.constraint](orig_x, args.poison_eps, args.step_size)

    if args.poison_type == 'Adv':
        target = (target + 1) % args.num_classes  # Error-maximizing noise: Using a fixed permutation of labels
    elif args.poison_type == 'Hyp':
        target = target  # Error-minimizing noise

    for _ in range(args.poison_steps):
        x = x.clone().detach().requires_grad_(True)
        if args.poison_aug == True:
            x_aug = trans(x)
            logits = model(x_aug)
        else:
            logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        grad = torch.autograd.grad(loss, [x])[0]
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
            x = torch.clamp(x, 0, 1)

    return x.clone().detach().requires_grad_(False)


def crafting_poison(args, loader, model):
    poisoned_input = []
    clean_target = []
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        inp, target = inp.cuda(), target.cuda()
        inp_p = batch_poison(model, inp, target, args)
        poisoned_input.append(inp_p.detach().cpu())
        clean_target.append(target.detach().cpu())
        with torch.no_grad():
            logits = model(inp_p)
            loss = nn.CrossEntropyLoss()(logits, target)
            acc = accuracy_top1(logits, target)
        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))
        desc = ('[{} {:.3f}] | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(args.poison_name, args.poison_eps, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)
    poisoned_input = torch.cat(poisoned_input, dim=0)
    clean_target = torch.cat(clean_target, dim=0)
    return poisoned_input, clean_target


def craft_poison_random(args, loader, model):
    # Generate random perturbations for each class
    poisons = torch.zeros(args.num_classes, *args.data_shape).cuda(non_blocking=True)
    step = STEPS[args.constraint](None, args.eps, None)
    poisons = step.random_perturb(poisons)
    poisoned_data = poison_random(args, loader, model, poisons)
    return poisoned_data


def poison_random(args, loader, model, poisons):
    poisoned_input = []
    clean_target = []
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Add the same perturbation to examples from the same class
        index = target.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).expand(-1, 3, 32, 32)
        delta = torch.gather(poisons, dim=0, index=index)

        inp_p = inp + delta
        inp_p = torch.clamp(inp_p, 0, 1)

        poisoned_input.append(inp_p.detach().cpu())
        clean_target.append(target.detach().cpu())

        with torch.no_grad():
            logits = model(inp_p)
            loss = nn.CrossEntropyLoss()(logits, target)
            acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))
        desc = ('[{} {}] | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(args.poison_type, args.eps, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)

    poisoned_input = torch.cat(poisoned_input, dim=0)
    clean_target = torch.cat(clean_target, dim=0)
    return poisoned_input, clean_target


def main(args):
    set_seed(args.seed)
    if os.path.isfile(args.poison_path):
        print('Poison [{}] already exists.'.format(args.poison_path))
        return
    if args.dataset == "CIFAR10":
        train_set = datasets.CIFAR10(args.data_path, train=True, transform=transforms.ToTensor())
        test_set = datasets.CIFAR10(args.data_path, train=False, transform=transforms.ToTensor())
    elif args.dataset == "CIFAR100":
        train_set = datasets.CIFAR100(args.data_path, train=True, transform=transforms.ToTensor())
        test_set = datasets.CIFAR100(args.data_path, train=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model, _ = make_and_restore_model(args.craft_model_arch, args.dataset, resume_path=args.model_path)
    model.eval()
    if args.poison_type == "Adv" or "Hyp":
        poison_data = crafting_poison(args, train_loader, model)
        torch.save(poison_data, args.poison_path)
        poison_data_test = crafting_poison(args, test_loader, model)
        torch.save(poison_data_test, args.poison_test_path)
    elif args.poison_type == "Random":
        poison_data = crafting_poison_random(args, train_loader, model)
        torch.save(poison_data, args.poison_path)
        poison_data_test = craft_poison_random(args, test_loader, model)
        torch.save(poison_data_test, args.poison_test_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generating poisons for CIFAR10 and CIFAR100')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out_dir', default='./results', type=str)
    parser.add_argument('--dataset', type=str, default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100"],
                        help='choose the dataset')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--poison_eps', default=8, type=float)
    parser.add_argument('--constraint', default='Linf', type=str, choices=['Linf', 'L2'])
    parser.add_argument('--poison_type', default='Hyp', type=str, choices=['Hyp', 'Adv', 'Random'])
    parser.add_argument('--poison_steps', default=100, type=int)
    parser.add_argument('--poison_aug', action='store_true',
                        help='if select, generate poison with data augmentation')

    parser.add_argument('--craft_model_loss', default='ST', type=str, choices=['ST', 'AT'])
    parser.add_argument('--craft_model_eps', default=2, type=float)
    parser.add_argument('--craft_model_epoch', default=150, type=int)
    parser.add_argument('--craft_model_arch', default='ResNet18', type=str)
    args = parser.parse_args()

    # Crafting options
    args.poison_eps = args.poison_eps / 255
    args.craft_model_eps = args.craft_model_eps / 255
    args.step_size = args.poison_eps / 10
    args.batch_size = 256

    args.data_path = os.path.join('./datasets', args.dataset)
    args.out_dir = os.path.join(args.out_dir, args.dataset)
    args.exp_name = infer_exp_name(args.craft_model_loss, args.craft_model_eps, args.craft_model_epoch,
                                   args.craft_model_arch, 'Clean', args.seed)
    args.model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pth')
    args.poison_name = infer_poison_name(args.poison_type, args.poison_steps, args.craft_model_loss,
                                         args.craft_model_eps, args.craft_model_epoch, args.craft_model_arch,
                                         args.poison_aug, args.poison_eps)
    args.poison_path = os.path.join(args.out_dir, args.exp_name, args.poison_name + '.poison')
    args.poison_test_path = os.path.join(args.out_dir, args.exp_name, args.poison_name + '.poison_test')

    pprint(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main(args)
