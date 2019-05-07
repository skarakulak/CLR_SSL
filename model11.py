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
from model_def11 import *

# reference: https://github.com/pytorch/examples/blob/master/imagenet/main.py


def train(
    sup_loader, unsup_loader,
    model, criterion, optimizer,
    gan_criterion, netG, optimizerG, netD, optimizerD,
    epoch, args, device, log_path
    ):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    loss_cse_meter = AverageMeter('Loss_cse', ':.4e')
    loss_cse_fake_meter = AverageMeter('Loss_cse_fake', ':.4e')
    loss_kld_s_meter = AverageMeter('Loss_kld_sup', ':.4e')
    loss_kld_us_meter = AverageMeter('Loss_kld_unsup', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    loss_d_x = AverageMeter('d_x', ':.4e')
    loss_d_xr = AverageMeter('d_xr', ':.4e')
    loss_d_z = AverageMeter('d_z', ':.4e')
    loss_g_z = AverageMeter('g_z', ':.4e')
    progress = ProgressMeter(
        len(sup_loader), batch_time, data_time, 
        losses, loss_cse_meter,loss_cse_fake_meter, 
        loss_kld_s_meter,loss_kld_us_meter,
        top1,top5, loss_d_x, loss_d_xr, loss_d_z, loss_g_z, 
        prefix="Epoch: [{}]".format(epoch)
    )


    # until epoch 45, we set `model.cl_centers` by sampling latent representations from the training examples
    # and make sure that we sample evenly among classes.
    model.eval()
    with torch.no_grad():
        if epoch <= 45:
            latent_reps_count = torch.zeros(1000)
            idx_cl=0
            for input_sup,y in sup_loader:
                input_sup = input_sup.to(device)
                output_sup, latent_sup, logvar_sup, c_dist_sup, cluster_sup  = model(input_sup, add_eps=False, return_c_dist=True)
                for (z,label) in zip(latent_sup,y):
                    if latent_reps_count[int(label)] <= torch.mean(latent_reps_count) + 1e-10:
                        model.cl_centers[idx_cl,:] = z.detach().data.clone()
                        idx_cl += 1
                        latent_reps_count[int(label)] += 1
                        if idx_cl>=args.num_of_clusters: break
                if idx_cl>=args.num_of_clusters: break
            model.cl_centers = nn.parameter.Parameter(model.cl_centers.data)




    # switch to train mode
    model.train()

    if epoch < 30: kld_multiplier = 0
    elif epoch < 40: kld_multiplier = 1e-4 * args.kld_multiplier
    elif epoch < 50: kld_multiplier = 1e-2 * args.kld_multiplier
    elif epoch < 60: kld_multiplier = 1e-1 * args.kld_multiplier
    else: kld_multiplier = args.kld_multiplier




    end = time.time()
    for i, ((input_sup, target_sup), (input_unsup,target_unsup)) in enumerate(zip(sup_loader,unsup_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        input_sup = input_sup.to(device)
        target_sup = target_sup.to(device)
        input_unsup = input_sup.to(device)
        target_unsup = target_sup.to(device)
        

        # until epoch 45, we set `model.cl_centers` by sampling latent representations from the training examples
        # and make sure that we sample evenly among classes.
        if i == 0 and epoch <= 45:
            latent_reps_count = torch.zeros(1000)
            idx_cl=0
            for input_sup,y in sup_loader:
                input_sup = input_sup.to(device)
                with torch.no_grad():
                    output_sup, latent_sup, logvar_sup, c_dist_sup, cluster_sup  = model(input_sup, add_eps=False, return_c_dist=True)
                for (z,label) in zip(latent_sup,y):
                    if latent_reps_count[int(label)] <= torch.mean(latent_reps_count) + 1e-10:
                        model.cl_centers[idx_cl,:] = z.detach().data.clone()
                        idx_cl += 1
                        latent_reps_count[int(label)] += 1
                        if idx_cl>=args.num_of_clusters: break
                if idx_cl>=args.num_of_clusters: break
            model.cl_centers = nn.parameter.Parameter(model.cl_centers.data)


        # compute output
        output_sup, mu_sup, logvar_sup, c_dist_sup, cluster_sup = model(input_sup, add_eps=True, return_c_dist=True)
        output_unsup, mu_unsup, logvar_unsup, c_dist_unsup, cluster_unsup  = model(input_unsup, add_eps=True, return_c_dist=True)
        
        # train the discriminator with the real data
        netD.zero_grad()
        batch_size = input_unsup.size(0)
        label = torch.full((batch_size,1), 1, device=device) #real label
        output = netD(input_unsup, model.cl_centers[cluster_unsup].detach())
        errD_real = gan_criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train the discriminator with the real data and random cluster IDs
        label.fill_(0)
        rand_cl_idx = torch.randint(high=args.num_of_clusters,size=(3,batch_size),device=device)
        output = netD(input_unsup, model.cl_centers[rand_cl_idx[0,:]].detach())
        errD_real = gan_criterion(output, label)
        errD_real.backward()
        D_xr = output.mean().item()


        # train the discriminator with the generated data (labels are still zero)
        z = model.cl_centers[rand_cl_idx[1,:]].detach() + torch.randn(32,512, device=device)
        fake = netG(z[:,:,None,None])
        output = netD(fake.detach(),model.cl_centers[rand_cl_idx[1,:]].detach())
        errD_fake = gan_criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()


        # update the generator
        netG.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        output = netD(fake,model.cl_centers[rand_cl_idx[2,:]].detach())
        errG = gan_criterion(output, label)
        errG.backward(retain_graph=True)
        D_G_z2 = output.mean()
        optimizerG.step()

        # generate examples for the resnet model
        with torch.no_grad():
            z = model.cl_centers[cluster_sup].detach() + torch.randn(32,512, device=device)
        input_fake = netG(z[:,:,None,None]).detach()
        output_fake, mu_fake, logvar_fake, c_dist_fake, cluster_fake  = model(input_fake, add_eps=True, return_c_dist=True)

        # update the resnet model. 
        loss_cse = criterion(output_sup, target_sup)
        loss_cse_fake = criterion(output_fake, target_sup)
        KLD_sup = -0.5 * torch.sum(1 + logvar_sup - c_dist_sup.pow(2) - logvar_sup.exp())
        KLD_unsup = -0.5 * torch.sum(1 + logvar_unsup - c_dist_unsup.pow(2) - logvar_unsup.exp())
        #KLD_fake = -0.5 * torch.sum(1 + logvar_fake - c_dist_fake.pow(2) - logvar_fake.exp())
        loss = loss_cse + kld_multiplier * ((8/9)*KLD_unsup+(1/9)*KLD_sup) + args.fake_cse_multiplier * loss_cse_fake

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_sup, target_sup, topk=(1, 5))
        losses.update(loss.item(), input_sup.size(0))
        loss_cse_meter.update(loss_cse.item(), input_sup.size(0))
        loss_cse_fake_meter.update(loss_cse_fake.item(), input_sup.size(0))
        loss_kld_s_meter.update(KLD_sup.item(), input_sup.size(0))
        loss_kld_us_meter.update(KLD_unsup.item(), input_sup.size(0))
        top1.update(acc1[0], input_sup.size(0))
        top5.update(acc5[0], input_sup.size(0))
        loss_d_x.update(D_x, input_sup.size(0))
        loss_d_xr.update(D_xr, input_sup.size(0))
        loss_d_z.update(D_G_z1, input_sup.size(0))
        loss_g_z.update(D_G_z2.item(), input_sup.size(0))

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
            output_val, mu_val, logvar_val = model(input_val, add_eps=False)
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
    load_cpoint_path_d = join(cpoint_folder_path,f'checkpoint_{args.weights_version_load}_d.pth.tar')
    save_cpoint_path_d = join(cpoint_folder_path,f'checkpoint_{args.weights_version_save}_d.pth.tar')
    load_cpoint_path_g = join(cpoint_folder_path,f'checkpoint_{args.weights_version_load}_g.pth.tar')
    save_cpoint_path_g = join(cpoint_folder_path,f'checkpoint_{args.weights_version_save}_g.pth.tar')

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
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(gan_weights_init)
    netD.apply(gan_weights_init)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    gan_criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(.5, 0.999))


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
    if isfile(load_cpoint_path_g):
        write_to_log(log_path,f' => loading checkpoint {args.weights_version_load}_g')
        checkpoint = torch.load(load_cpoint_path_g)    
        netG.load_state_dict(checkpoint['state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizer'])
        write_to_log(log_path,f' => loaded checkpoint {args.weights_version_load}_g')
    if isfile(load_cpoint_path_d):
        write_to_log(log_path,f' => loading checkpoint {args.weights_version_load}_d')
        checkpoint = torch.load(load_cpoint_path_d)    
        netD.load_state_dict(checkpoint['state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizer'])
        write_to_log(log_path,f' => loaded checkpoint {args.weights_version_load}_d')

    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        #adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(
            data_loader_sup_train, data_loader_unsup,
            model, criterion, optimizer,
            gan_criterion, netG, optimizerG, netD, optimizerD,
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



