 
 
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
## few clinets on mobilenet
 
 

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
    
    assert args.local_upt_part in ['body', 'head', 'full'] #and args.aggr_part in ['body', 'head', 'full']
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

     # loading the net_tutor
    net_tutor = get_model(args)
    base_dir_tutor = '../save/OURS/{}/{}_U{}_S{}_F{}_lp{}/{}/'.format(args.dataset, args.model, args.num_users, args.shard_per_user, args.frac, args.local_ep, 'final')
    net_tutor_path = os.path.join(base_dir_tutor, algo_dir, 'best_model.pt')
    print('load model: ',net_tutor_path)
    net_tutor.load_state_dict(torch.load(net_tutor_path), strict=True) 



    # build local models
    net_local_tutor_list = []
    net_local_tutor_list.append(copy.deepcopy(net_tutor))

    
    # fine-tuning
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')
    results = []
    best_acc_tutor   = None
    lr = args.lr
    ol = args.ol                                           # DJ
    personalization_epoch = 5

    before_acc_results = np.zeros(args.num_users)
    after_acc_results  = np.zeros(args.num_users)

    ##  Finetune round  #####################################################################################################################
    # Update  -----------------------------------------------------------------------------
    for idx in range(args.num_users):
        local_tutor = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])  # dj local_tutor model training setting
        # net_local_tutor = copy.deepcopy(net_local_tutor_list[idx])                               # dj local_tutor model definition 
        net_local_tutor = copy.deepcopy(net_local_tutor_list[0])                               # dj local_tutor model definition 

        # breakpoint()
        acc_test, loss_test = test_img_local(net_local_tutor, dataset_test, -1, args, user_idx=idx, idxs=dict_users_test[idx])
        # acc_test, loss_test = test_img_local(net_local_tutor, dataset_test, 0, args, user_idx=idx, idxs=dict_users_test[idx])
        # acc_test, loss_test = test_img_local(net_local_tutor, dataset_test, 1, args, user_idx=idx, idxs=dict_users_test[idx])
        # acc_test, loss_test = test_img_local(net_local_tutor, dataset_test, 2, args, user_idx=idx, idxs=dict_users_test[idx])
        before_acc_results[idx] = acc_test

        # breakpoint()
        # tic_train = datetime.now()
        if args.local_upt_part == 'body':
            weight_tutor, loss_tutor = local_tutor.train_MHKD(net=net_local_tutor.to(args.device), body_lr=lr, head_lr=0., Temporature=args.temperature, KD_weight=1005, out_layer=ol, local_eps=personalization_epoch)
        if args.local_upt_part == 'head':
            weight_tutor, loss_tutor = local_tutor.train_MHKD(net=net_local_tutor.to(args.device), body_lr=0., head_lr=lr, Temporature=args.temperature, KD_weight=1005, out_layer=ol, local_eps=personalization_epoch)    
        if args.local_upt_part == 'full':
            weight_tutor, loss_tutor = local_tutor.train_MHKD(net=net_local_tutor.to(args.device), body_lr=lr, head_lr=lr, Temporature=args.temperature, KD_weight=1005, out_layer=ol, local_eps=personalization_epoch)
        
        net_local_tutor.load_state_dict(weight_tutor, strict=False)

        # breakpoint()
        acc_test_after, loss_test_after = test_img_local(net_local_tutor, dataset_test, -1, args, user_idx=idx, idxs=dict_users_test[idx])
        # acc_test, loss_test = test_img_local(net_local_tutor, dataset_test, 0, args, user_idx=idx, idxs=dict_users_test[idx])
        # acc_test, loss_test = test_img_local(net_local_tutor, dataset_test, 1, args, user_idx=idx, idxs=dict_users_test[idx])
        # acc_test, loss_test = test_img_local(net_local_tutor, dataset_test, 2, args, user_idx=idx, idxs=dict_users_test[idx])
        after_acc_results[idx] = acc_test_after 

        print('After FINE TUNING, USER {:3d}: Test accuracy from {:>6.2f} to {:>6.2f}'.format(idx+1, acc_test, acc_test_after))

    ACC_before_FT = np.mean(before_acc_results)
    STD_before_FT = np.std(before_acc_results)
    ACC_after_FT = np.mean(after_acc_results)
    STD_after_FT = np.std(after_acc_results)

    print('FINE TUNING: Test accuracy {:>6.2f} to {:>6.2f}, Test std {:>5.2f} to {:>5.2f}'.format(ACC_before_FT, ACC_after_FT, STD_before_FT, STD_after_FT))
 
    results.append(np.array([5, ACC_before_FT, STD_before_FT, ACC_after_FT, STD_after_FT]))
    final_results = np.array(results)
    final_results = pd.DataFrame(final_results, columns=['finetune', 'ACC_before_FT', 'STD_before_FT', 'ACC_after_FT', 'STD_after_FT'])
    final_results.to_csv(results_save_path, index=False)

