import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy_top1
from attacks.natural import natural_attack
from attacks.adv import adv_attack, batch_adv_attack
from attacks.trades import batch_trades_attack


def standard_loss(model, x, y):
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    return loss, logits


def adv_loss(args, model, x, y):
    model.eval()
    x_adv = batch_adv_attack(args, model, x, y)
    model.train()

    logits_adv = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits_adv, y)
    return loss, logits_adv


LOSS_FUNC = {
    '': standard_loss,
    'ST': standard_loss,
    'AT': adv_loss,
}


def train(args, model, optimizer, loader, writer, epoch):
    model.train()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()

    iterator = tqdm(enumerate(loader), total=len(loader), ncols=95)
    for i, (inp, target) in iterator:
        inp = inp.cuda()
        target = target.cuda()

        loss, logits = LOSS_FUNC[args.train_loss](args, model, inp, target)
        acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        desc = 'Train Epoch: {} | Loss {:.4f} | Accuracy {:.4f} ||'.format(epoch, loss_logger.avg, acc_logger.avg)
        iterator.set_description(desc)

    if writer is not None:
        descs = ['loss', 'accuracy']
        vals = [loss_logger, acc_logger]
        for d, v in zip(descs, vals):
            writer.add_scalar('train_{}'.format(d), v.avg, epoch)

    return loss_logger.avg, acc_logger.avg


def train_model(args, model, optimizer, train_loader, test_loader, writer, resume=0, schedule=None):
    """no schedule"""
    if args.epochs == 0:
        checkpoint = {
            'model': model.state_dict(),
            'epoch': 0,
            'train_acc': -1,
            'train_loss': -1,
            'nat_clean_train_acc': -1,
            'nat_clean_test_acc': -1,
            'adv_clean_train_acc': -1,
            'adv_clean_test_acc': -1,
        }
        torch.save(checkpoint, args.model_path)
        torch.save(checkpoint, args.model_path_last)

    for epoch in range(resume+1, args.epochs+1):
        train(args, model, optimizer, train_loader, writer, epoch)

        last_epoch = (epoch == (args.epochs - 1))
        should_log = (epoch % args.log_gap == 0)

        if should_log or last_epoch:
            nat_clean_train_loss, nat_clean_train_acc, _ = natural_attack(
                args, model, train_loader, writer, epoch, 'clean_train')
            nat_clean_test_loss, nat_clean_test_acc, _ = natural_attack(
                args, model, test_loader, writer, epoch, 'clean_test')

            robust_target = (args.train_loss in ['AT', 'TRADES', 'MART'])
            if robust_target:
                adv_clean_train_loss, adv_clean_train_acc, _ = adv_attack(
                    args, model, train_loader, writer, epoch, 'clean_train')
                adv_clean_test_loss, adv_clean_test_acc, _ = adv_attack(
                    args, model, test_loader, writer, epoch, 'clean_test')
            else:
                adv_clean_train_loss, adv_clean_train_acc, adv_clean_test_loss, adv_clean_test_acc = -1, -1, -1, -1

            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'train_acc': -1,
                'train_loss': -1,
                'nat_clean_train_acc': nat_clean_train_acc,
                'nat_clean_test_acc': nat_clean_test_acc,
                'adv_clean_train_acc': adv_clean_train_acc,
                'adv_clean_test_acc': adv_clean_test_acc,
            }
            torch.save(checkpoint, args.model_path)
            torch.save(checkpoint, args.model_path_last)
        if schedule:
            schedule.step()
    return model


def poison_train_model(args, model, optimizer, poison_train_loader, clean_test_loader, clean_train_loader,
                       poison_test_loader, writer, resume=0):
    if args.epochs == 0:
        checkpoint = {
            'model': model.state_dict(),
            'epoch': 0,
            'train_acc': -1,
            'train_loss': -1,
            'nat_clean_train_loss': -1,
            'nat_clean_test_loss': -1,
            'nat_poison_train_acc': -1,
            'nat_poison_test_acc': -1,
            'adv_clean_train_acc': -1,
            'adv_clean_test_acc': -1,
            'adv_poison_train_acc': -1,
            'adv_poison_test_acc': -1,
        }
        torch.save(checkpoint, args.model_path)
        torch.save(checkpoint, args.model_path_last)

    for epoch in range(resume+1, args.epochs+1):
        train_loss, train_acc = train(args, model, optimizer, poison_train_loader, writer, epoch)

        last_epoch = (epoch == (args.epochs - 1))
        should_log = (epoch % args.log_gap == 0)

        if should_log or last_epoch:
            nat_clean_train_loss, nat_clean_train_acc, _ = natural_attack(
                args, model, clean_train_loader, writer, epoch, 'clean_train')
            nat_clean_test_loss, nat_clean_test_acc, _ = natural_attack(
                args, model, clean_test_loader, writer, epoch, 'clean_test')
            nat_poison_train_loss, nat_poison_train_acc, _ = natural_attack(
                args, model, poison_train_loader, writer, epoch, 'poison_train')
            nat_poison_test_loss, nat_poison_test_acc, _ = natural_attack(
                args, model, poison_test_loader, writer, epoch, 'poison_test')

            robust_target = (args.train_loss in ['AT', 'TRADES', 'MART'])
            if robust_target:
                adv_clean_train_loss, adv_clean_train_acc, _ = adv_attack(
                    args, model, clean_train_loader, writer, epoch, 'clean_train')
                adv_clean_test_loss, adv_clean_test_acc, _ = adv_attack(
                    args, model, clean_test_loader, writer, epoch, 'clean_test')
                adv_poison_train_loss, adv_poison_train_acc, _ = adv_attack(
                    args, model, poison_train_loader, writer, epoch, 'poison_train')
                adv_poison_test_loss, adv_poison_test_acc, _ = adv_attack(
                     args, model, poison_test_loader, writer, epoch, 'poison_test')

            else:
                adv_clean_train_acc, adv_clean_test_acc = -1, -1
                adv_poison_train_acc, adv_poison_test_acc = -1, -1

            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'nat_clean_train_acc': nat_clean_train_acc,
                'nat_clean_test_acc': nat_clean_test_acc,
                'nat_poison_train_acc': nat_poison_train_acc,
                'nat_poison_test_acc': nat_poison_test_acc,
                'adv_clean_train_acc': adv_clean_train_acc,
                'adv_clean_test_acc': adv_clean_test_acc,
                'adv_poison_train_acc': adv_poison_train_acc,
                'adv_poison_test_acc': adv_poison_test_acc,
            }
            torch.save(checkpoint, args.model_path)
            torch.save(checkpoint, args.model_path_last)
    return model


def eval_model(args, model, loader):
    model.eval()
    args.eps = args.eps

    keys, values = [], []
    keys.append('Model')
    values.append(args.model_path)

    # Natural
    _, acc, name = natural_attack(args, model, loader)
    keys.append(name)
    values.append(acc)

    # FGSM
    args.num_steps = 1
    args.step_size = args.eps
    args.random_restarts = 0
    _, acc, name = adv_attack(args, model, loader)
    keys.append('FGSM')
    values.append(acc)

    # PGD-10
    args.num_steps = 10
    args.step_size = args.eps / 4
    args.random_restarts = 1
    _, acc, name = adv_attack(args, model, loader)
    keys.append(name)
    values.append(acc)

    # PGD-20
    args.num_steps = 20
    args.step_size = args.eps / 4
    args.random_restarts = 1
    _, acc, name = adv_attack(args, model, loader)
    keys.append(name)
    values.append(acc)

    # PGD-100
    # args.num_steps = 100
    # args.step_size = args.eps / 4
    # args.random_restarts = 1
    # _, acc, name = adv_attack(args, model, loader)
    # keys.append(name)
    # values.append(acc)

    # CW-100
    # from attacks.cw import cw_attack
    # args.num_steps = 100
    # args.step_size = args.eps / 4
    # args.random_restarts = 1
    # _, acc, name = cw_attack(args, model, loader)
    # keys.append(name)
    # values.append(acc)

    # AutoAttack
    # from autoattack import AutoAttack
    # adversary = AutoAttack(model, norm=args.constraint, eps=args.eps, version='standard')
    # x_test = torch.cat([x for (x, y) in loader])
    # y_test = torch.cat([y for (x, y) in loader])
    # x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
    # auto_acc = adversary.clean_accuracy(x_adv, y_test, bs=args.batch_size) * 100
    # keys.append('AotuAttack')
    # values.append(auto_acc)

    # Save results
    import csv
    csv_fn = '{}.csv'.format(args.model_path)
    with open(csv_fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(keys)
        write.writerow(values)

    print('=> csv file is saved at [{}]'.format(csv_fn))

