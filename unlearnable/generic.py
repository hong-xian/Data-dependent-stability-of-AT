import pickle
import os
import sys
import logging
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import data
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import models


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean

    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


def get_transforms(dataset, train=True, is_tensor=True):
    if train:
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            comp1 = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4), ]
        else:
            raise NotImplementedError
    else:
        comp1 = []

    if is_tensor:
        comp2 = [
            transforms.Normalize((255 * 0.5, 255 * 0.5, 255 * 0.5), (255., 255., 255.))
         ]
    else:
        comp2 = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))
            ]

    trans = transforms.Compose([*comp1, *comp2])

    if is_tensor:
        trans = data.ElementWiseTransform(trans)

    return trans


def get_dataset(dataset, root='./data', train=True):
    transform = get_transforms(dataset, train=train, is_tensor=False)
    if dataset == 'CIFAR10':
        target_set = data.datasetCIFAR10(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    elif dataset == 'CIFAR100':
        target_set = data.datasetCIFAR100(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    return data.Dataset(x, y, transform)


def get_concat_dataset(dataset, root='./data'):
    transform = get_transforms(dataset, train=True, is_tensor=False)
    if dataset == 'CIFAR10':
        train_set = data.datasetCIFAR10(root=root, train=True, transform=transform)
        test_set = data.datasetCIFAR10(root=root, train=False, transform=transform)
        x_1, y_1 = train_set.data, train_set.targets
        x_2, y_2 = test_set.data, test_set.targets
        x = np.concatenate([x_1, x_2])
        y = np.concatenate([y_1, y_2])

    elif dataset == 'CIFAR100':
        train_set = data.datasetCIFAR100(root=root, train=True, transform=transform)
        test_set = data.datasetCIFAR100(root=root, train=False, transform=transform)
        x_1, y_1 = train_set.data, train_set.targets
        x_2, y_2 = test_set.data, test_set.targets
        x = np.concatenate([x_1, x_2])
        y = np.concatenate([y_1, y_2])
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    return data.Dataset(x, y, transform)


def get_concat_indexed_loader(dataset, batch_size, root='./data'):
    target_set = get_concat_dataset(dataset, root=root)
    target_set = data.IndexedDataset(x=target_set.x, y=target_set.y, transform=target_set.transform)
    loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader


def get_concat_indexed_tensor_loader(dataset, batch_size, root='./data'):
    target_set = get_concat_dataset(dataset, root=root)
    target_set = data.IndexedTensorDataset(x=target_set.x, y=target_set.y)
    loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader


def get_indexed_loader(dataset, batch_size, root='./data', train=True):
    target_set = get_dataset(dataset, root=root, train=train)

    if train:
        target_set = data.IndexedDataset(x=target_set.x, y=target_set.y, transform=target_set.transform)
    else:
        target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False)

    return loader


def get_indexed_tensor_loader(dataset, batch_size, root='./data', train=True):
    target_set = get_dataset(dataset, root=root, train=train)
    target_set = data.IndexedTensorDataset(x=target_set.x, y=target_set.y)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=False)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_arch(arch, dataset):
    if dataset == "CIFAR10":
        if arch == 'ResNet18':
            model = models.ResNet18()
        elif arch == 'VGG16':
            model = models.VGG('VGG16')
        elif arch == 'WRN28-10':
            model = models.WideResNet(depth=28, num_classes=10, widen_factor=10)
        else:
            raise NotImplementedError('architecture {} is not supported'.format(arch))
    elif dataset == "CIFAR100":
        if arch == 'ResNet18':
            model = models.ResNet18(num_classes=100)
        elif arch == 'VGG16':
            model = models.VGG('VGG16', num_classes=100)
        elif arch == 'WRN28-10':
            model = models.WideResNet(depth=28, num_classes=100, widen_factor=10)
        else:
            raise NotImplementedError('architecture {} is not supported'.format(arch))
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))
    return model


def get_optim(optim, params, lr=0.1, weight_decay=1e-4, momentum=0.9):
    if optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    raise NotImplementedError('optimizer {} is not supported'.format(optim))


def generic_init(args):
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        '{}/{}_log.txt'.format(args.save_dir, args.save_name), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    logger.info('Arguments')
    for arg in vars(args):
        logger.info('    {:<22}        {}'.format(arg + ':', getattr(args, arg)))
    logger.info('')

    return logger


def evaluate(model, criterion, loader):
    acc = AverageMeter()
    loss = AverageMeter()

    model.eval()
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            _y = model(x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(x)
            lo = criterion(_y, y).item()
        acc.update(ac, len(x))
        loss.update(lo, len(x))

    return acc.average(), loss.average()


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_model_state(model):
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model.state_dict()
    return model_state


def concat_get_poisoned_loader(
        dataset, batch_size, root='./data', train=True,
        noise_path=None):
    target_set = get_dataset(dataset, root=root, train=train)
    if noise_path is not None:
        with open(noise_path, 'rb') as f:
            raw_noise = pickle.load(f)

        assert isinstance(raw_noise, np.ndarray)
        assert raw_noise.dtype == np.int8

        raw_noise = raw_noise.astype(np.int16)
        ''' restore noise (NCWH) for raw images (NHWC) '''
        noise = np.transpose(raw_noise, [0, 2, 3, 1])
        ''' add noise to images (uint8, 0~255) '''
        if train:
            noise = noise[:len(target_set)]
            imgs = target_set.x.astype(np.int16) + noise
            imgs = imgs.clip(0, 255).astype(np.uint8)
            target_set.x = imgs
        else:
            noise = noise[-len(target_set):]
            imgs = target_set.x.astype(np.int16) + noise
            imgs = imgs.clip(0, 255).astype(np.uint8)
            target_set.x = imgs
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.ToTensor()

    if train:
        target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=transform_train)
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, num_workers=8)
    else:
        target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=transform_test)
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, num_workers=8)

    return loader


def get_poisoned_loader(dataset, batch_size, root='./data', noise_path=None):
    train_loader = concat_get_poisoned_loader(dataset, batch_size, root, True, noise_path)
    test_loader = concat_get_poisoned_loader(dataset, batch_size, root, False, noise_path)
    return train_loader, test_loader
