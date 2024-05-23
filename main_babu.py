#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img, test_img_local, test_img_local_all
import os

import pdb

import random                 # DJ
from datetime import datetime # D
from torch.utils.tensorboard import SummaryWriter # DJ



## variants ---------------------------------------------------------------------------------------
## full means complete network; rltK means network using K-th output
## for evaluating KD (for main_bkd_babu.py)
## strong teachers --------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol 12 --results_save full
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  7 --results_save full
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol 12 --results_save full
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  7 --results_save full
## weak teachers ----------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol 12 --results_save full
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol 12 --results_save full
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol 12 --results_save full
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  7 --results_save full
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  7 --results_save full
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  7 --results_save full

## strong students --------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 1.0 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
## weak students ----------------------------------------------------------------------------------
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_babu.py --dataset cifar100 --model mobile   --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  3 --results_save rlt3
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users 100 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5
# python main_babu.py --dataset cifar100 --model resnet18 --num_classes 100 --shard_per_user 100 --epochs 320 --lr 0.1 --num_users  10 --frac 0.5 --local_ep 1 --local_bs 50 --local_upt_part body --aggr_part body --momentum 0.90 --wd 0.0 --seed 1 --gpu 2 --temperature 5 --KD_weight 10 --KD_mode 0 --ol  5 --results_save rlt5


# launch tensorboard
# CUDA_VISIBLE_DEVICES=0 tensorboard --logdir=/home/djchen/Projects/FederatedLearning/save/BABU --port 1231

# dj
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
    
    # Fix random Seed
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(args.seed)
    make_deterministic(args.seed)
    
    assert args.local_upt_part in ['body', 'head', 'full'] and args.aggr_part in ['body', 'head', 'full']
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.unbalanced:
        base_dir = '../save/{}/{}_iid{}_num{}_C{}_le{}_m{}_wd{}/shard{}_sdr{}_unbalanced_bu{}_md{}/{}/'.format(
            args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.momentum, args.wd, args.shard_per_user, args.server_data_ratio, args.num_batch_users, args.moved_data_size, args.results_save)
    else:
        base_dir = '../save/{}/{}_iid{}_num{}_C{}_le{}_m{}_wd{}/shard{}_sdr{}/{}/'.format(
            args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.momentum, args.wd, args.shard_per_user, args.server_data_ratio, args.results_save)
    # algo_dir = 'local_upt_{}_aggr_{}'.format(args.local_upt_part, args.aggr_part)
    
    ### brief path anme # dj
    base_dir = '../save/BABU/{}/{}_U{}_C{}_S{}/{}/'.format(
            args.dataset, args.model, args.num_users, args.frac, args.shard_per_user, args.results_save)
    algo_dir = ''  
    
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    dict_users_train = {int(k): np.array(v, dtype=int) for k, v in dict_users_train.items()}
    dict_users_test = {int(k): np.array(v, dtype=int) for k, v in dict_users_test.items()}
    
    dict_save_path = os.path.join(base_dir, algo_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    # build a global model
    net_glob = get_model(args)
    net_glob.train()


    # breakpoint()
    # build local models
    net_local_list = []
    net_local_list.append(copy.deepcopy(net_glob))
    
    # training
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')
    results = []
    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    ol = args.ol                                           # DJ
    lr_schedule = [(args.epochs*2)//4, (args.epochs*3)//4] # dj original setting
    args.test_freq = 5                                     # testing frequency
    log_dir = os.path.join(base_dir, algo_dir)             # DJ
    swriter = SummaryWriter(log_dir)                       # DJ
    tic = datetime.now()
    for iter in range(args.epochs):
        w_glob = None
        loss_locals = []
        
        # Client Sampling
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))
        
        # Local Updates
        for idx in idxs_users:
            # print(idx)
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_local_list[0])
            
            tic_train = datetime.now()
            if args.local_upt_part == 'body':
                w_local, loss = local.train(net=net_local.to(args.device), body_lr=lr, head_lr=0., out_layer=ol)
            if args.local_upt_part == 'head':
                w_local, loss = local.train(net=net_local.to(args.device), body_lr=0., head_lr=lr, out_layer=ol)
            if args.local_upt_part == 'full':
                w_local, loss = local.train(net=net_local.to(args.device), body_lr=lr, head_lr=lr, out_layer=ol)
            toc_train = datetime.now()
            print(iter,idx, '  Elapsed training time: %.3f seconds.' % ((toc_train-tic_train).total_seconds()))
                
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
        
        # Aggregation
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)
        
        # Broadcast
        update_keys = list(w_glob.keys())
        if args.aggr_part == 'body':
            if args.server_data_ratio > 0.0:
                pass
            else:
                update_keys = [k for k in update_keys if 'linear' not in k]
        elif args.aggr_part == 'head':
            update_keys = [k for k in update_keys if 'linear' in k]
        elif args.aggr_part == 'full':
            pass
        w_glob = {k: v for k, v in w_glob.items() if k in update_keys}
        net_local_list[0].load_state_dict(w_glob, strict=False)
        
        if (iter + 1) in lr_schedule:
            lr *= 0.1

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            # acc_test, loss_test = test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=False)
            acc_test, loss_test = test_img(net_local_list[0], dataset_test, args, out_layer=ol)

            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter
                best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
                torch.save(net_local_list[0].state_dict(), best_save_path)

            results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)
            swriter.add_scalar('ACC-Round', acc_test, iter+1)  # DJ 
            # swriter.add_scalar('Loss-Round', loss_avg, iter+1) # DJ 
    swriter.close()      # DJ   
    toc = datetime.now() # DJ

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
    print('Elapsed time: %f seconds.' % (toc-tic).total_seconds())