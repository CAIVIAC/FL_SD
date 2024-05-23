#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime # DJ
from utils.options import args_parser
from utils.train_utils import get_data, get_model
from torch.utils.tensorboard import SummaryWriter # DJ
import random                 # DJ

## baselines --------------------------------------------------------------------------------------
## one user for evaluating KD (for main_bkd_sng.py)
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol 12 --results_save full
# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  7 --results_save full

## variants --------------------------------------------------------------------------------------
## full means complete network; rltK means network using K-th output
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  2 --results_save rlt2
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  3 --results_save rlt3
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  4 --results_save rlt4
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  5 --results_save rlt5
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  6 --results_save rlt6
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  7 --results_save rlt7
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  8 --results_save rlt8
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  9 --results_save rlt9
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol 10 --results_save rlt10
# python main_single.py --dataset cifar100 --model mobile   --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol 11 --results_save rlt11

# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  2 --results_save rlt2
# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  3 --results_save rlt3
# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  4 --results_save rlt4
# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  5 --results_save rlt5
# python main_single.py --dataset cifar100 --model resnet18 --num_classes 100 --local_upt_part body --aggr_part body --epochs 320 --shard_per_user 100 --num_users 1 --frac 1.0 --local_ep 1 --seed 1 --gpu 0 --ol  6 --results_save rlt6


# launch tensorboard
# CUDA_VISIBLE_DEVICES=0 tensorboard --logdir=/home/djchen/Projects/FederatedLearning/save/SNG --port 2341



def make_deterministic(seed):
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    seed=int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True,warn_only=True)
    # torch.autograd.set_detect_anomaly(True) # for debug, slowing the process

    os.environ["PYTHONHASHSEED"] = str(seed) # for hash function
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # modify the function _worker_init_fn defined in /home/djchen/PROJECTS/HumanNeRF/superhumannerf/core/data/create_dataset.py
    # see https://pytorch.org/docs/stable/notes/randomness.html#pytorch

if __name__ == '__main__':
    # parse args
    args = args_parser()
    make_deterministic(args.seed)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    base_dir = '../save/SNG/{}/{}_U{}_C{}_S{}/{}/'.format(
            args.dataset, args.model, args.num_users, args.frac, args.shard_per_user, args.results_save)    
    algo_dir = ''  

    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    # set dataset
    dataset_train, dataset_test = get_data(args, env='single')
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=128, num_workers=4)
    dataloaders = {'train': train_loader, 'test': test_loader}

    # build a model
    net_glob = get_model(args)
    total_params_tutor = "{:,}".format(sum(p.numel() for p in net_glob.parameters()))
    print(f"net_glob parameters: {total_params_tutor}")
    
    # Basically, He uniform
    if args.results_save=='xavier_uniform':
        nn.init.xavier_uniform_(net_glob.linear.weight, gain=nn.init.calculate_gain('relu'))
    elif args.results_save=='xavier_normal':
        nn.init.xavier_normal_(net_glob.linear.weight, gain=nn.init.calculate_gain('relu'))
    elif args.results_save=='kaiming_uniform':
        nn.init.kaiming_uniform_(net_glob.linear.weight, nonlinearity='relu')
    elif args.results_save=='kaiming_normal':
        nn.init.kaiming_normal(net_glob.linear.weight, nonlinearity='relu')
    elif args.results_save=='orthogonal':
        nn.init.orthogonal_(net_glob.linear.weight, gain=nn.init.calculate_gain('relu'))
    elif args.results_save=='not_orthogonal':
        nn.init.uniform_(net_glob.linear.weight, a=0.45, b=0.55)
        net_glob.linear.weight.data = net_glob.linear.weight.data / torch.norm(net_glob.linear.weight.data, dim=1, keepdim=True)

    # nn.init.zeros_(net_glob.linear.bias) #   do: 61 
                                         # undo:
        
    # set optimizer
    body_params = [p for name, p in net_glob.named_parameters() if 'linear' not in name]
    head_params = [p for name, p in net_glob.named_parameters() if 'linear' in name]
    
    args.body_lr, args.head_lr = args.lr, args.lr                          # DJ
    args.body_m, args.head_m   = args.momentum, args.momentum              # DJ
    print(args.body_lr, args.head_lr, args.body_m, args.head_m )
    # breakpoint()
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': args.body_lr, 'momentum': args.body_m},
                                     {'params': head_params, 'lr': args.head_lr, 'momentum': args.head_m}],
                                     weight_decay=5e-4)
    elif args.opt == 'RMSProp':
        optimizer = torch.optim.RMSprop([{'params': body_params, 'lr': args.body_lr, 'momentum': args.body_m},
                                         {'params': head_params, 'lr': args.head_lr, 'momentum': args.head_m}],
                                          weight_decay=5e-4)
    elif args.opt == 'ADAM':
        optimizer = torch.optim.Adam([{'params': body_params, 'lr': args.body_lr, 'betas': (args.body_m, 1.11*args.body_m)},
                                      {'params': head_params, 'lr': args.head_lr, 'betas': (args.head_m, 1.11*args.head_m)}],
                                       weight_decay=5e-4)
    
    # set scheduler    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     [args.epochs//2, (args.epochs*3)//4],
                                                     gamma=0.1,
                                                     last_epoch=-1)

    
    # set criterion
    criterion = nn.CrossEntropyLoss()
    
    # training
    results_log_save_path = os.path.join(base_dir, algo_dir, 'results.csv')
    results_model_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
            
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    args.test_freq = 5                                                     # testing frequency
    log_dir = os.path.join(base_dir, algo_dir)                             # DJ
    swriter = SummaryWriter(log_dir)                                       # DJ
    out_layer = args.ol                                                    # DJ
    onlyTrainBatches = 50 # 50 #############################
    for epoch in tqdm(range(args.epochs)):
        net_glob.train()
        train_loss = 0
        train_correct = 0
        train_data_num = 0
        batch_loss = [] # DJ

        tic_here = datetime.now()
        for i, data in enumerate(dataloaders['train']):
            # if i > onlyTrainBatches: #########
            #     break                #########
            
            image = data[0].type(torch.FloatTensor).to(args.device)
            label = data[1].type(torch.LongTensor).to(args.device)
            
            pred_label = net_glob(image, out_layer)                        # DJ
            # pred_label = net_glob(image)                        # DJ
            loss = criterion(pred_label, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item()) # DJ

            pred_label = torch.argmax(pred_label, dim=1)
            train_loss += loss.item()
            train_correct += (torch.sum(pred_label==label).item())
            train_data_num += label.shape[0]
            # print(i, label)
        toc_here = datetime.now()
        # print(epoch, 'Elapsed time: %.3f seconds.' % ((toc_here-tic_here).total_seconds()))    
        # print(batch_loss)
        # breakpoint()

        if (epoch + 1) % args.test_freq == 0:
            net_glob.eval()
            test_loss = 0
            test_correct = 0
            test_data_num = 0
            for i, data in enumerate(dataloaders['test']):
                image = data[0].type(torch.FloatTensor).to(args.device)
                label = data[1].type(torch.LongTensor).to(args.device)

                pred_label = net_glob(image, out_layer)                    # DJ  
                # pred_label = net_glob(image)                    # DJ  
                loss = criterion(pred_label, label)
                
                pred_label = torch.argmax(pred_label, dim=1)
                test_loss += loss.item()
                test_correct += (torch.sum(pred_label==label).item())
                test_data_num += label.shape[0]
                # print(i, label)
            # breakpoint()
            # print(epoch, 'Correct images: %6d' % (test_correct))
                
            train_loss_list.append(train_loss/len(dataloaders['train']))
            train_acc_list.append(train_correct/train_data_num)
            test_loss_list.append(test_loss/len(dataloaders['test']))
            test_acc_list.append(test_correct/test_data_num)
            
            res_pd = pd.DataFrame(data=np.array([train_loss_list, train_acc_list, test_loss_list, test_acc_list]).T,
                                columns=['train_loss', 'train_acc', 'test_loss', 'test_acc'])
            res_pd.to_csv(results_log_save_path, index=False)
            if (test_correct/test_data_num) >= max(test_acc_list):
                torch.save(net_glob.state_dict(), results_model_save_path)
                
            
            acc_test = test_correct/test_data_num*100                          # DJ
            # swriter.add_scalars('ACC-Round', {'Single': acc_test}, epoch+1)  # DJ
            swriter.add_scalar('ACC-Round', acc_test, epoch+1)                 # DJ

        scheduler.step()    
    swriter.close()                                                            # DJ