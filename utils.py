import os
from os.path import isfile, join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import shutil

# references
# https://github.com/pytorch/examples/blob/master/imagenet/main.py



def safe_mkdir(path):
    "Creates a directory if there isn't one already."
    try:
        os.mkdir(path)
    except OSError:
        pass


def write_to_log(log_path,str_to_log):
    with open(log_path ,'a') as lgfile:
        lgfile.write(str_to_log + '\n')
        lgfile.flush()


def image_loader(path, batch_size, num_workers=3, pin_memory = True, valid_crop = 84):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5011, 0.4727, 0.4229), (0.2269, 0.2223, 0.2258))
        ]
    )
    if valid_crop:
        transform_valid = transforms.Compose(
            [
                transforms.CenterCrop(valid_crop),
                transforms.ToTensor(), 
                transforms.Normalize((0.5011, 0.4727, 0.4229), (0.2269, 0.2223, 0.2258))
            ]
        )
    else:
        transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5011, 0.4727, 0.4229), (0.2269, 0.2223, 0.2258))
            ]
        )

    sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform_train)
    sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform_valid)
    unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=transform_train)
    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    data_loader_unsup = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return data_loader_sup_train, data_loader_sup_val, data_loader_unsup


def find_mean_and_variance(data_loaders):
    '''
    returns mean of variance of the images in the given
    list of dataloaders
    '''
    mean = 0.
    std = 0.
    nb_samples = 0.
    for dl in data_loaders:
        for data,y in dl:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples


    mean /= nb_samples
    std /= nb_samples
    return(mean,std)


def save_checkpoint(state, is_best, cpoint_folder_path, version = 'v0'):
    torch.save(state, join(cpoint_folder_path,f'checkpoint_{version}.pth.tar'))
    if is_best:
        shutil.copyfile(
            join(cpoint_folder_path,f'checkpoint_{version}.pth.tar'), 
            join(cpoint_folder_path,f'checkpoint_{version}_best.pth.tar')
        )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch,log_path):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        write_to_log(log_path,'\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def gan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)