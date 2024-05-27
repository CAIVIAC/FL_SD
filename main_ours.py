





#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model, count_layer_parameters
from models.Update import LocalUpdate
from models.test import test_img, test_img_local, test_img_local_all
import os

import pdb

import random                 # DJ
from datetime import datetime # DJ
import time
from torch.utils.tensorboard import SummaryWriter # DJ


#### OURS: Multi-Head KD --------------------------------------------------------------------------
 

# launch tensorboard
# CUDA_VISIBLE_DEVICES=0 tensorboard --logdir=/home/djchen/Projects/FederatedLearning/save/OURS --port 1234



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
    base_dir = '../save/OURS/{}/{}_U{}_S{}_F{}_lp{}/{}/'.format(
            args.dataset, args.model, args.num_users, args.shard_per_user, args.frac, args.local_ep, args.results_save)
    algo_dir = ''    

    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    # dict_users_train = {int(k): np.array(v, dtype=int) for k, v in dict_users_train.items()} # redundant
    # dict_users_test1 = {int(k): np.array(v, dtype=int) for k, v in dict_users_test.items()}  # redundant
 
    # reduce the training data amount for faster training toy example
    if False:
        data_size =  dict_users_train[0].shape[0]
        keep_ratio = 0.1
        for u in range(args.num_users):
            dict_users_train[u] = dict_users_train[u][1:int(data_size*keep_ratio)]
            # print(u, dict_users_train[u].shape[0])
    if False:
        data_size =  dict_users_test[0].shape[0]
        keep_ratio = 0.1
        for u in range(args.num_users):
            dict_users_test[u] = dict_users_test[u][1:int(data_size*keep_ratio)]
            # print(u, dict_users_test[u].shape[0])

    dict_save_path = os.path.join(base_dir, algo_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    # build a tutor model
    net_tutor = get_model(args)
    net_tutor.train()
    body_params_tutor = sum(p.numel() for name, p in net_tutor.named_parameters() if 'linear' not in name)
    body_params_tutor_str = "{:,}".format(body_params_tutor)
    print(f"Tutor body parameters: {body_params_tutor_str}")
    head_params_tutor = "{:,}".format(sum(p.numel() for name, p in net_tutor.named_parameters() if 'linear' in name))
    print(f"    Tutor head parameters: {head_params_tutor}")
    head_names_tutor = [name for name, p in net_tutor.named_parameters() if 'linear' in name]
    # print(f"    Tutor head names: {head_names_tutor}")
 
    # build a tutee model
    net_tutee = get_model(args)
    net_tutee.train()
    # count_model_parameters(net_tutee) #
    layer_list = [name for name, p in net_tutee.named_parameters() if 'alpha' not in name]
    ending_idx = 3 + 6*args.ol
    body_params_tutee = count_layer_parameters(net_tutee,layer_list[:ending_idx])
    body_params_tutee_str = "{:,}".format(body_params_tutee)
    print(f"Tutee body parameters: {body_params_tutee_str}")
    print(f"    Tutee body names: {layer_list[:ending_idx]}")
    print("-  "*10 + "RATIO of TuteeBody/TutorBody: " + "{:.3f}".format(body_params_tutee/body_params_tutor) + "  -"*10)
    

    # build local models
    net_local_tutor_list = []
    # for user_idx in range(args.num_users):
    net_local_tutor_list.append(copy.deepcopy(net_tutor))
    net_local_tutee_list = []
    # for user_idx in range(args.num_users):
    net_local_tutee_list.append(copy.deepcopy(net_tutee))
    
    # training
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')
    results = []
    best_acc_tutor   = None
    best_acc_tutee   = None
    best_epoch_tutor = None
    best_epoch_tutee = None

    lr = args.lr
    ol = args.ol                                           # DJ
    lr_schedule = [(args.epochs*2)//4, (args.epochs*3)//4] # dj original setting
    args.test_freq = 8                                     # testing frequency
    log_dir = os.path.join(base_dir, algo_dir)             # DJ
    swriter = SummaryWriter(log_dir)                       # DJ
    tic_start = datetime.now()
    for iter in range(args.epochs):
        tic_epoch = datetime.now()
        w_glob_tutor  = None
        w_glob_tutee  = None
        loss_locals_tutor = []
        loss_locals_tutee = []
        
        # Client Sampling
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("Round {}, lr: {:.6f}, {}, Time {}".format(iter, lr, idxs_users,time.strftime("%H:%M:%S", time.localtime())))
        
        ##  TUTOR round  #####################################################################################################################
        # if iter < args.epochs/2 #or (iter >= args.epochs/2 and iter%2==0) : # FL training 
        # Update  -----------------------------------------------------------------------------
        for idx in idxs_users:
            local_tutor = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])  # dj local_tutor model training setting
            # net_local_tutor = copy.deepcopy(net_local_tutor_list[idx])                               # dj local_tutor model definition 
            net_local_tutor = copy.deepcopy(net_local_tutor_list[0])                               # dj local_tutor model definition 

            # tic_train = datetime.now()
            if args.local_upt_part == 'body':
                weight_tutor, loss_tutor = local_tutor.train_MHKD(net=net_local_tutor.to(args.device), body_lr=lr, head_lr=0., Temporature=args.temperature, KD_weight=args.KD_weight, out_layer=ol)
            elif args.local_upt_part == 'full':
                weight_tutor, loss_tutor = local_tutor.train_MHKD(net=net_local_tutor.to(args.device), body_lr=lr, head_lr=lr, Temporature=args.temperature, KD_weight=args.KD_weight, out_layer=ol)
            # toc_train = datetime.now()
            # print(iter,idx, '  Elapsed tutor training time: %.3f seconds.' % ((toc_train-tic_train).total_seconds()))    
                    
            loss_locals_tutor.append(copy.deepcopy(loss_tutor))

            if w_glob_tutor is None:
                w_glob_tutor = copy.deepcopy(weight_tutor)
            else:
                for k in w_glob_tutor.keys():
                    w_glob_tutor[k] += weight_tutor[k]

        # Aggregation -----------------------------------------------------------------------------
        for k in w_glob_tutor.keys():
            w_glob_tutor[k] = torch.div(w_glob_tutor[k], m)

        # Broadcast -------------------------------------------------------------------------------
        update_keys = list(w_glob_tutor.keys())
        if args.aggr_part == 'body':
            if args.server_data_ratio > 0.0:
                pass
            else:
                update_keys = [k for k in update_keys if 'linear' not in k]
        elif args.aggr_part == 'full':
            pass
        w_glob_tutor = {k: v for k, v in w_glob_tutor.items() if k in update_keys}
        # for user_idx in range(args.num_users):
        #     net_local_tutor_list[user_idx].load_state_dict(w_glob_tutor, strict=False)
        net_local_tutor_list[0].load_state_dict(w_glob_tutor, strict=False)    

 


        if (iter + 1) in lr_schedule:
            lr *= 0.1

        # print loss
        loss_avg_tutor = sum(loss_locals_tutor) / len(loss_locals_tutor)
        # loss_avg_tutee = sum(loss_locals_tutee) / len(loss_locals_tutee)
 
        if (iter + 1) % args.test_freq == 0:
            # tic_eval = datetime.now()
            acc_test_tutor,  loss_test_tutor,  acc_test_tutor_std  = test_img_local_all(net_local_tutor_list, args, dataset_test, dict_users_test, out_layer=-1, return_all=False)
            acc_test_tutee0, loss_test_tutee0, acc_test_tutee0_std = test_img_local_all(net_local_tutor_list, args, dataset_test, dict_users_test, out_layer= 0, return_all=False)
            acc_test_tutee1, loss_test_tutee1, acc_test_tutee1_std = test_img_local_all(net_local_tutor_list, args, dataset_test, dict_users_test, out_layer= 1, return_all=False)
            acc_test_tutee2, loss_test_tutee2, acc_test_tutee2_std = test_img_local_all(net_local_tutor_list, args, dataset_test, dict_users_test, out_layer= 2, return_all=False)
            # acc_test_tutor, loss_test_tutor = test_img(net_local_tutor_list[0], dataset_test, args, out_layer=-1)
            # acc_test_tutee, loss_test_tutee = test_img(net_local_tutor_list[0], dataset_test, args, out_layer=ol)
            # breakpoint()
            # toc_eval = datetime.now()
            # print('Evaluation time: %.3f seconds.' % ((toc_eval-tic_eval).total_seconds()))
            print('ROUND {:3d}: Test accuracy (Tutor/Tutee0/Tutee1/Tutee2) {:>6.2f}/{:>6.2f}/{:>6.2f}/{:>6.2f}, Test std (Tutor/Tutee0/Tutee1/Tutee2) {:>5.2f}/{:>5.2f}/{:>5.2f}/{:>5.2f}, Time {}'.format(iter+1, acc_test_tutor, acc_test_tutee0, acc_test_tutee1, acc_test_tutee2, acc_test_tutor_std, acc_test_tutee0_std, acc_test_tutee1_std, acc_test_tutee2_std, time.strftime("%H:%M:%S", time.localtime())))
            
            if best_acc_tutor is None or acc_test_tutor > best_acc_tutor:
                best_acc_tutor = acc_test_tutor
                best_epoch_tutor = iter
                best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
                torch.save(net_local_tutor_list[0].state_dict(), best_save_path) # dj: should be an aggregated model

            # if best_acc_tutee is None or acc_test_tutee > best_acc_tutee:
            #     best_acc_tutee = acc_test_tutee
            #     best_epoch_tutee = iter
            #     best_save_path = os.path.join(base_dir, algo_dir, 'best_tutee_model.pt')
            #     torch.save(net_local_tutee_list[0].state_dict(), best_save_path) # dj: should be an aggregated model

            results.append(np.array([iter, acc_test_tutor, acc_test_tutee0, acc_test_tutee1, acc_test_tutee2, best_acc_tutor, acc_test_tutor_std, acc_test_tutee0_std, acc_test_tutee1_std, acc_test_tutee2_std]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'acc_test_tutor', 'acc_test_tutee0', 'acc_test_tutee1', 'acc_test_tutee2', 'best_acc_tutor', 'acc_test_tutor_std', 'acc_test_tutee0_std', 'acc_test_tutee1_std', 'acc_test_tutee2_std'])
            final_results.to_csv(results_save_path, index=False)
            swriter.add_scalars('ACC-Round', {'Tutor': acc_test_tutor}, iter+1)  # DJ
            swriter.add_scalars('ACC-Round', {'Tutee0': acc_test_tutee0}, iter+1)  # DJ     
            swriter.add_scalars('ACC-Round', {'Tutee1': acc_test_tutee1}, iter+1)  # DJ  
            swriter.add_scalars('ACC-Round', {'Tutee2': acc_test_tutee2}, iter+1)  # DJ  
            # swriter.add_scalars('Loss-Round', {'Tutor': loss_avg_tutor,'Tutee': loss_avg_tutee}, iter+1) # DJ             
    swriter.close()      # DJ
    toc_end = datetime.now()
    print('Best model, iter: {}, acc: {}'.format(best_epoch_tutor, best_acc_tutor))
    print('Elapsed time: %.3f minutes.' % ((toc_end-tic_start).total_seconds()/60))