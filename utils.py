import os
from os.path import isfile, join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import shutil
import PIL
import math
import random
import socket

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
        lgfile.write(f'{str_to_log} ({socket.gethostname()}) \n')
        lgfile.flush()


def image_loader(path, batch_size, num_workers=3, pin_memory = True, valid_crop = 84, subset_n=64):
    transform_train = transforms.Compose(
        [   
            #transforms.RandomRotation(12, resample=PIL.Image.BILINEAR),
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

    if subset_n == 64:
        data_loader_sup_train = torch.utils.data.DataLoader(
            sup_train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        train_ind = sum([list(i*64+np.random.choice(64, subset_n, replace=False)) for i in range(1000)],[])
        train_sampler = SubsetRandomSampler(train_ind)
        data_loader_sup_train = torch.utils.data.DataLoader(
            sup_train_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler
        )


    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data,
        batch_size=batch_size,
        shuffle=False,
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
def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data].cuda()

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot_embedding(target, input.size(-1))
        inp = input.clamp(self.eps, 1. - self.eps)

        loss = -1 * Variable(y) * torch.log(inp) # cross entropy
        loss = loss * (1 - inp) ** self.gamma # focal loss
        return loss.sum(dim=1).mean()


def init_clusters(model, sup_loader, unsup_loader,device, args, num_of_classes=1000):
    num_cl_lim = math.ceil(args.num_of_clusters/num_of_classes)
    model.eval()
    with torch.no_grad():
        latent_reps_count = torch.zeros(num_of_classes)
        idx_cl=0
        for input_sup,y in sup_loader:
            input_sup = input_sup.to(device)
            output_sup, latents = model(input_sup, return_latent=True)
            for (z1,z2,z3,label) in zip(*latents,y):
                if latent_reps_count[int(label)] < num_cl_lim:
                    for cl,z in zip(model.cl_centers,[z1,z2,z3]):
                        if z.ndimension()>2:
                            rh,rw = random.randint(0,z.size(1)-1), random.randint(0,z.size(2)-1)
                            cl[idx_cl,:] = z[:,rh,rw].detach().data.clone()
                        else: cl[idx_cl,:] = z.detach().data.clone()
                    idx_cl += 1
                    latent_reps_count[int(label)] += 1
                    if idx_cl>=args.num_of_clusters: break
            if idx_cl>=args.num_of_clusters: break
        for cl in model.cl_centers: cl = nn.parameter.Parameter(cl.data)


def get_label_hierarchy(path):
    tree_labels_path = {}
    tree_label_full_path = {}
    tree_paths = set()
    with open(path,'r') as f:
        for line in f:
            label, l_path = line.split(',')[:2]
            full_path = l_path.strip()
            tree_label_full_path[int(label)]=[int(l) for l in list(full_path)]
            path_labels ={}
            for k in range(1,1+len(full_path)):
                tree_paths.add(full_path[:k-1])
                path_labels[full_path[:k-1]] = int(full_path[k-1])
            tree_labels_path[int(label)] = path_labels
    path_inds = {k:i for i,k in enumerate(sorted(tree_paths,key=len))}
    tree_labels_path_indexed = {
        l:{path_inds[p]:p_l for p,p_l in path_dict.items()} 
        for l, path_dict in tree_labels_path.items()
    }
    labels_hier_idx = {}
    for k, v in tree_labels_path_indexed.items():
        idx,labs = list(zip(*v.items()))
        labels_hier_idx[k] = (list(idx),list(labs))
    return labels_hier_idx, len(tree_paths), path_inds

def pred_path_with_threshold(pred, path_inds, start_ind, p_threshold=.66):
    current_node=0
    node_cumul_prob = 1.
    current_path = []
    cur_node_path_idx = [0]
    while True:     
        next_path_pred = pred[cur_node_path_idx[-1]]
        next_path_prob = max(next_path_pred,1-next_path_pred)
        node_cumul_prob *= next_path_prob
        if node_cumul_prob<p_threshold: 
            break
        else: 
            current_path.append('1' if next_path_pred >= .5 else '0')
            new_path = ''.join(current_path)
            if new_path not in path_inds: 
                return [start_ind+i for i in cur_node_path_idx]

            cur_node_path_idx.append(path_inds[new_path])
    return [start_ind+i for i in cur_node_path_idx[:-1]]


def pred_class_hsfmx(pred, path_inds, start_ind):
    current_node=0
    current_path = []
    cur_node_path_idx = [0]
    while True:     
        next_path_pred = pred[cur_node_path_idx[-1]]
        next_path_prob = max(next_path_pred,1-next_path_pred)
        node_cumul_prob *= next_path_prob
        if node_cumul_prob<p_threshold: 
            break
        else: 
            current_path.append('1' if next_path_pred >= .5 else '0')
            new_path = ''.join(current_path)
            if new_path not in path_inds: 
                return [start_ind+i for i in cur_node_path_idx]

            cur_node_path_idx.append(path_inds[new_path])
    return [start_ind+i for i in cur_node_path_idx[:-1]]
