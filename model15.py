import os
from os.path import isfile, join
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import random
from collections import defaultdict
from utils import *
from model_def15 import *

# reference: https://github.com/pytorch/examples/blob/master/imagenet/main.py


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



def train(
    sup_loader, unsup_loader,
    model, criterion, optimizer,
    epoch, args, device, log_path
    ):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    loss_cse_meter = AverageMeter('Loss_cse', ':.4e')
    loss_cdist_s_meter = AverageMeter('Loss_cdist_sup', ':.4e')
    loss_cdist_us_meter = AverageMeter('Loss_cdist_unsup', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(sup_loader), batch_time, data_time, 
        losses, loss_cse_meter,
        loss_cdist_s_meter,loss_cdist_us_meter,
        top1,top5,
        prefix="Epoch: [{}]".format(epoch)
    )

    # until epoch 55, we set `model.cl_centers` by sampling latent representations from the training examples
    # and make sure that we sample evenly among classes.
    if epoch <= 55:
        init_clusters(model, sup_loader, unsup_loader, device, args)

    # switch to train mode
    model.train()
    
    if args.cdist_loss_schedule and epoch < 65:
        if epoch < 40: cdist_multiplier = 0
        elif epoch < 50: cdist_multiplier = 1e-3 * args.cdist_multiplier
        elif epoch < 65: cdist_multiplier = 1e-1 * args.cdist_multiplier
    else: cdist_multiplier = args.cdist_multiplier
    

    end = time.time()
    for i, ((input_sup, target_sup), (input_unsup,target_unsup)) in enumerate(zip(sup_loader,unsup_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        input_sup = input_sup.to(device)
        target_sup = target_sup.to(device)
        input_unsup = input_sup.to(device)
        target_unsup = target_sup.to(device)
        
        # compute output
        output_sup, c_dist_sup = model(input_sup, return_c_dist=True)
        output_unsup, c_dist_unsup  = model(input_unsup, return_c_dist=True)

        # update the resnet model. 
        loss_cse = criterion(output_sup, target_sup)
        loss = loss_cse + cdist_multiplier * ((8/9)*c_dist_sup+(1/9)*c_dist_unsup) 


        acc1, acc5 = accuracy(output_sup, target_sup, topk=(1, 5))
        losses.update(loss.item(), input_sup.size(0))
        loss_cse_meter.update(loss_cse.item(), input_sup.size(0))
        loss_cdist_s_meter.update(c_dist_sup.item(), input_sup.size(0))
        loss_cdist_us_meter.update(c_dist_unsup.item(), input_sup.size(0))
        top1.update(acc1[0], input_sup.size(0))
        top5.update(acc5[0], input_sup.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.print(i,log_path)



def validate(val_loader, model, criterion, args,device, log_path):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader), batch_time, losses,top1, top5, prefix='Test: '
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_val, target_val) in enumerate(val_loader):
            input_val = input_val.to(device)
            target_val = target_val.to(device)

            # compute output
            output_val = model(input_val)
            loss = criterion(output_val, target_val)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output_val, target_val, topk=(1, 5))
            losses.update(loss.item(), input_val.size(0))
            top1.update(acc1[0], input_val.size(0))
            top5.update(acc5[0], input_val.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % args.print_freq == 0:
                progress.print(i,log_path)

        # TODO: this should also be done with the ProgressMeter
    write_to_log(
        log_path,
        ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
    )
        
    return top1.avg


def train_and_val(args):
    cudnn.benchmark = True
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.hpc == 'prince':
        safe_mkdir(f'/scratch/{args.user}/dl_competition/')
        cpoint_folder_path = f'/scratch/{args.user}/dl_competition/checkpoints/'
        data_path = f'/home/{args.user}/unsupervised_learning_competition/ssl_data_96'
    else:
        safe_mkdir(f'/data/{args.user}/dl_competition/')
        cpoint_folder_path = f'/data/{args.user}/dl_competition/checkpoints/'
        data_path = f'/data/{args.user}/ssl_data_96'

    safe_mkdir(cpoint_folder_path)
    
    load_cpoint_path = join(cpoint_folder_path,f'checkpoint_{args.weights_version_load}.pth.tar')
    save_cpoint_path = join(cpoint_folder_path,f'checkpoint_{args.weights_version_save}.pth.tar')

    safe_mkdir('../logs')
    log_path = f'../logs/log_{args.version}.txt'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    data_loader_sup_train, data_loader_sup_val, data_loader_unsup = image_loader(
        data_path,32,num_workers=args.num_of_workers, valid_crop = None
    )

    write_to_log(log_path, '\n'.join([f'{key}: {value}' for key,value in vars(args).items()])+'\n\n' )

    global best_acc1
    # create model
    if args.arch=='resnet32':
        model = resnet34(
            num_clust = args.num_of_clusters) 
    else:
        model = resnet18(
            num_clust = args.num_of_clusters)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()


    if args.set_optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # torch.optim.SGD(model.parameters(), args['lr'],
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if isfile(load_cpoint_path):
        write_to_log(log_path,f' => loading checkpoint {args.weights_version_load}')
        checkpoint = torch.load(load_cpoint_path)
        if checkpoint['arch'] != args.arch:
            write_to_log(log_path,f' ===> model architecture saved at checkpoint {args.weights_version_load} is different.')
            return
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
    
        model.load_state_dict(checkpoint['state_dict'])
        if checkpoint['optimizer_name'] == args.set_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            write_to_log(log_path,f' ===> loaded optimizer state for {args.set_optimizer}')
        write_to_log(log_path,f' => loaded checkpoint {args.weights_version_load}')
    else:
        best_acc1 = -1


    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        #adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(
            data_loader_sup_train, data_loader_unsup,
            model, criterion, optimizer,
            epoch, args, device, log_path
        )

        # evaluate on validation set
        acc1 = validate(data_loader_sup_val, model, criterion, args, device, log_path)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'optimizer_name' : args.set_optimizer
            },
            is_best,
            cpoint_folder_path,
            args.weights_version_save
        )



