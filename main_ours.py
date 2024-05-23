# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Python version: 3.6

# import copy
# import pickle
# import numpy as np
# import pandas as pd
# import torch

# from utils.options import args_parser
# from utils.train_utils import get_data, get_model, count_model_parameters, count_layer_parameters, update_scheme, get_layer_list
# from models.Update import LocalUpdate
# from models.test import test_img, test_img_local, test_img_local_all
# import os

# import pdb

# import random                 # DJ
# from datetime import datetime # DJ
# from scipy.special import softmax
# from torch.utils.tensorboard import SummaryWriter # DJ

# # dj
# def make_deterministic(seed):
#     '''Seed everything for better reproducibility.
#     (some pytorch operation is non-deterministic like the backprop of grid_samples)
#     '''
#     seed=int(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed) # sets the seed for generating random numbers.
#     torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
#     torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
#     # torch.backends.cudnn.enabled = False
#     # torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     # torch.use_deterministic_algorithms(True,warn_only=True)
#     # torch.autograd.set_detect_anomaly(True) # for debug, slowing the process

#     os.environ["PYTHONHASHSEED"] = str(seed) # for hash function
#     os.environ["CUDA_LAUNCH_BLOCKING"] = str(seed)
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#     # modify the function _worker_init_fn defined in /home/djchen/PROJECTS/HumanNeRF/superhumannerf/core/data/create_dataset.py
#     # see https://pytorch.org/docs/stable/notes/randomness.html#pytorch

# if __name__ == '__main__':
#     # parse args
#     args = args_parser()
    
#     # Fix random Seed
#     make_deterministic(args.seed)

#     assert args.local_upt_part in ['body', 'head', 'full'] and args.aggr_part in ['body', 'head', 'full']
#     args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

#     if args.unbalanced:
#         base_dir = '../save/{}_{}/{}_iid{}_num{}_C{}_le{}_m{}_wd{}/shard{}_sdr{}_unbalanced_bu{}_md{}/{}/'.format(
#             args.dataset, args.scheme, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.momentum, args.wd, args.shard_per_user, args.server_data_ratio, args.num_batch_users, args.moved_data_size, args.results_save)
#     else:
#         base_dir = '../save/{}_{}/{}_iid{}_num{}_C{}_le{}_m{}_wd{}/shard{}_sdr{}/{}/'.format(
#             args.dataset, args.scheme, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.momentum, args.wd, args.shard_per_user, args.server_data_ratio, args.results_save)
#     # algo_dir = 'local_upt_{}_aggr_{}'.format(args.local_upt_part, args.aggr_part)
    
#     ### brief path anme # dj
#     base_dir = '../save/OURS/{}/{}_U{}_C{}_S{}/{}/'.format(
#             args.dataset, args.model, args.num_users, args.frac, args.shard_per_user, args.results_save)
#     algo_dir = ''


#     if not os.path.exists(os.path.join(base_dir, algo_dir)):
#         os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

#     # DJ: each client gets different data
#     dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
#     dict_users_train = {int(k): np.array(v, dtype=int) for k, v in dict_users_train.items()}
#     dict_users_test = {int(k): np.array(v, dtype=int) for k, v in dict_users_test.items()}
#     # breakpoint()

#     dict_save_path = os.path.join(base_dir, algo_dir, 'dict_users.pkl')
#     with open(dict_save_path, 'wb') as handle:
#         pickle.dump((dict_users_train, dict_users_test), handle)

#     # building models
#     net_tutor, net_tutee = get_model(args) # net_tutee: client/server; net_tutor: client
#     net_tutor.train()
#     net_tutee.train()
#     # body_params = [p for name, p in net_tutee.named_parameters() if 'linear' not in name]  # dj

#     # build local models # DJ: each client uses the same model
#     net_local_tutor_list = []
#     for user_idx in range(args.num_users):
#         net_local_tutor_list.append(copy.deepcopy(net_tutor))
#     net_local_tutee_list = []
#     for user_idx in range(args.num_users):
#         net_local_tutee_list.append(copy.deepcopy(net_tutee))
    
#     # check models
#     tutor_layer_name_list = get_layer_list(net_tutor)
#     tutee_layer_name_list = get_layer_list(net_tutee)
#     if True:
#         count_model_parameters(net_tutor) # JC
#         print(tutor_layer_name_list)  # DJ
#         count_model_parameters(net_tutee) # JC
#         print(tutee_layer_name_list)  # DJ
#     # breakpoint()
#     ###### Define the update module #####################################################################
#     if args.model=='mobile':
#         # store the net layers layer-by-layer -----------------
#         layer_dict = {}
#         layer_dict['init'] = [name for name, p in net_tutee.named_parameters() if 'layers' not in name and 'linear' not in name]
#         print(f"{'init':<8} {'>>   Paras:'} {count_layer_parameters(net_tutee, layer_dict['init']):>7} {'   Names:':<5} {layer_dict['init']}")
#         search_key = [f'layers.{i}.' for i in range(13)]
#         # search_key = [f'layers.{i}.' for i in range(9)]
#         for k in range(len(search_key)):
#             layer_dict[f'layer{k}'] = [name for name, p in net_tutee.named_parameters() if search_key[k] in name]
#             print(f"{f'layer{k}':<8} {'>>   Paras:'} {count_layer_parameters(net_tutee, layer_dict[f'layer{k}']):>7} {'   Names:':<5} {layer_dict[f'layer{k}']}")
#         layer_dict['head'] = [name for name, p in net_tutee.named_parameters() if 'linear' in name]
#         print(f"{'head':<8} {'>>   Paras:'} {count_layer_parameters(net_tutee, layer_dict['head']):>7} {'   Names:':<5} {layer_dict['head']}")
#         M0  = layer_dict['init']
#         M1  = layer_dict[f'layer{0}']
#         M2  = layer_dict[f'layer{1}']
#         M3  = layer_dict[f'layer{2}']
#         M4  = layer_dict[f'layer{3}']
#         M5  = layer_dict[f'layer{4}']
#         M6  = layer_dict[f'layer{5}']
#         M7  = layer_dict[f'layer{6}']
#         M8  = layer_dict[f'layer{7}']
#         M9  = layer_dict[f'layer{8}']
#         M10 = layer_dict[f'layer{9}']
#         M11 = layer_dict[f'layer{10}']
#         M12 = layer_dict[f'layer{11}']
#         M13 = layer_dict[f'layer{12}']
#         module_pool = [M0,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13]   
#         # module_pool = [M0,M1,M2,M3,M4,M5,M6,M7,M8,M9]    
#         num_module = len(module_pool)
#         # breakpoint()
#     elif args.model=='resnet18':
#         # store the net layers layer-by-layer -----------------
#         layer_dict = {}
#         layer_dict['init'] = [name for name, p in net_tutee.named_parameters() if 'layer' not in name and 'linear' not in name]
#         print(f"{'init':<8} {'>>   Paras:'} {count_layer_parameters(net_tutee, layer_dict['init']):>7} {'   Names:':<5} {layer_dict['init']}")

#         search_key = [f'layer{i+1}' for i in range(4)]
#         for k in range(len(search_key)):
#             layer_dict[f'layer{k}'] = [name for name, p in net_tutee.named_parameters() if search_key[k] in name]
#             print(f"{f'layer{k}':<8} {'>>   Paras:'} {count_layer_parameters(net_tutee, layer_dict[f'layer{k}']):>7} {'   Names:':<5} {layer_dict[f'layer{k}']}")
        
#         layer_dict['head'] = [name for name, p in net_tutee.named_parameters() if 'linear' in name]
#         print(f"{'head':<8} {'>>   Paras:'} {count_layer_parameters(net_tutee, layer_dict['head']):>7} {'   Names:':<5} {layer_dict['head']}")
#         M0  = layer_dict['init']
#         M1  = layer_dict[f'layer{0}']
#         M2  = layer_dict[f'layer{1}']
#         M3  = layer_dict[f'layer{2}']
#         M4  = layer_dict[f'layer{3}']
#         module_pool = [M0,M1,M2,M3,M4]       
#         num_module = len(module_pool)
#         # print(get_layer_list(net_tutee))
#         # breakpoint()
#     else:
#         exit('Error: unrecognized model')        
#     # breakpoint()
#     ###########################################################################

#     # training
#     results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')
#     results = []
#     loss_train = []
#     net_best   = None
#     best_loss  = None
#     best_acc   = None
#     best_epoch = None

#     lr = args.lr
#     lr_schedule = [(args.epochs* 8)//16, (args.epochs*12)//16]
#     # lr_schedule = [(args.epochs*23)//32, (args.epochs*29)//32] # mobile: 34.75 res18: 51.01
#     # lr_schedule = [(args.epochs*24)//32, (args.epochs*28)//32] #               res18: 51.62
#     # lr_schedule = [(args.epochs*26)//32, (args.epochs*28)//32] #               res18: 51.53
#     # lr_schedule = [(args.epochs*30)//32]                       # mobile: 34.52 res18: 52.24
#     # lr_schedule = [(args.epochs*30)//32, (args.epochs*31)//32] # mobile: 34.14 res18: 51.64
#     # lr_schedule = [(args.epochs*26)//32, (args.epochs*29)//32] # mobile: 35.36 res18: 51.89
#     # lr_schedule = [(args.epochs*26)//32, (args.epochs*30)//32] # mobile: 35.25 res18: 52.52
#     # lr_schedule = [(args.epochs*27)//32, (args.epochs*30)//32] # mobile: 35.01 res18: 52.15
#     # lr_schedule = [(args.epochs*28)//32, (args.epochs*30)//32] # mobile: 35.00 res18: 52.26
#     # lr_schedule = [(args.epochs*25)//32, (args.epochs*30)//32] # mobile: 35.07 res18: 51.76

    
#     # get the updating schedule according to the scheme. 
#     # update_layers = update_scheme(args.epochs, args.scheme, args.warmup, lr_schedule)   # DJ define a updating list per epoch
#     # print(update_layers) # current layer-wise update not use module-wise update
#     # breakpoint()


#     tic = datetime.now()
#     num_users_per_epoch = max(int(args.frac * args.num_users), 1)
#     # define a client queue list
#     use_client_queue = True
#     if use_client_queue:
#         client_queue = []
#         # # case 1 each shuffle using all clients
#         # for i in range(int(np.ceil(num_users_per_epoch * args.epochs / args.num_users))):
#         #     client_shuffle = np.random.choice(range(args.num_users), args.num_users, replace=False)
#         #     client_queue = np.append(client_queue,client_shuffle,axis=0)
#         # client_queue = client_queue.astype(int)
#         # client_queue = client_queue.tolist()
#         # case 2 each shuffle using num_users_per_epoch clients
#         for i in range(int(num_users_per_epoch * args.epochs / num_users_per_epoch+1)):
#             client_shuffle = np.random.choice(range(args.num_users), num_users_per_epoch, replace=False)
#             client_queue = np.append(client_queue,client_shuffle,axis=0)
#         client_queue = client_queue.astype(int)
#         client_queue = client_queue.tolist()
#     toc = datetime.now()
#     print("queue shuffle use %f seconds" % (toc-tic).total_seconds())


#     tic = datetime.now() # DJ
#     log_dir = os.path.join(base_dir, algo_dir) # DJ
#     swriter = SummaryWriter(log_dir) # DJ
#     uniform_aggregation = True   # DJ
#     acc_test = 0 # DJ
#     loss_avg = 0 # DJ    
#     keep_ratio = args.kr # DJ
#     idxs_users = [] # DJ
#     loss_clients = np.inf * np.ones(args.num_users) # DJ
#     layer_update_cnt_all_epochs = dict(zip(tutee_layer_name_list, [0 for name in tutee_layer_name_list])) # DJ : counter for layer updating all epochs
#     for ep in range(args.epochs):
#         weight_global       = None
#         loss_local_pool     = [] # dj 
#         std_local_pool      = [] # dj
#         weight_local_pool   = [] # dj
#         layer_update_cnt_per_epoch = dict(zip(tutee_layer_name_list, [0 for name in tutee_layer_name_list])) # DJ : counter for layer updating one epoch
         
#         # clear clients last epoch
#         if ep%2==0  or keep_ratio==0 :
#             print("\n\n","- - - " * 20)
#             print("clear clients last 2 epochs: ", idxs_users)
#             idxs_users = []
#             num_kick_clients = int(num_users_per_epoch * (1-keep_ratio))
#         else:
#             print("- - - " * 10)

#         # updaters in this epoch
#         if use_client_queue and ep%2 == 0:
#             old_users = []
#             old_users.append([client_queue.pop(0) for idx in range(num_kick_clients)])            
#             new_users = []
#             new_users.append([client_queue.pop(0) for idx in range(num_kick_clients)])
#             idxs_users = old_users[0] + new_users[0]
#             print("[old clients] [new clients]: ", old_users[0], new_users[0])
#         elif use_client_queue and ep%2 == 1:
#             dummy = old_users        
#             old_users = new_users
#             new_users = dummy
#             idxs_users = old_users[0] + new_users[0]
#             print("[old clients] [new clients]: ", old_users[0], new_users[0])            
#         else: # random clients sampling
#             idxs_users = np.append(idxs_users,np.random.choice(range(args.num_users), num_kick_clients, replace=False),axis=0)
#             idxs_users = idxs_users.astype(int)
#         # print(idxs_users)

#         # skedule each layer updated by which updaters/clients
#         num_modules_per_updater = int(np.ceil(args.fb_ratio * num_module))  # each client update how many layers
#         num_updaters_per_module = int(args.fb_ratio * num_users_per_epoch)  # each layer updated by how many clients
#         layer_updater = []  
#         queue_updater = (old_users[0] + new_users[0]) * num_modules_per_updater
#         queue_updater = queue_updater[:num_updaters_per_module*num_module]        
#         for m in range(num_module):
#             layer_updater.append([queue_updater.pop(0) for idx in range(num_updaters_per_module)])
#         print("Updaters per layer per epoch:", layer_updater)

#         # Local Updates --------------------------------------------------------------------------------------------------------------
#         tic_local_update = datetime.now() # DJ
#         layer_updater_idx = []
#         for m in range(num_module):
#             layer_updater_idx.append([])
#         tutee_acc_pool = [] # dj: store test acc in KD training
#         tutor_acc_pool = [] # dj: store test acc in KD training
#         for ii, idx in enumerate(idxs_users):
#             # print(ii, idx)    
#             # print("Client {} in {}".format(idx, idxs_users)) # DJ

#             ##  TUTOR/TUTEE initialization  #########################################################################################################
#             local_tutor = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])  # dj local_tutor model training setting
#             net_local_tutor = copy.deepcopy(net_local_tutor_list[idx])                               # dj local_tutor model definition    
#             local_tutee = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])  # dj local_tutee model training setting
#             net_local_tutee = copy.deepcopy(net_local_tutee_list[idx])                               # dj local_tutee model definition
#             # Print the norm of the first layer of the new lightweight model
#             if ep==0 and ii==0:
#                 if args.model=='mobile':
#                     print("Norm of 1st layer of tutor:  ", torch.norm(net_local_tutor.layers[0].conv1.weight).item())
#                     print("Norm of 1st layer of tutee:  ", torch.norm(net_local_tutee.layers[0].conv1.weight).item())
#                 if args.model=='resnet18':
#                     print("Norm of 1st layer of tutor:  ", torch.norm(net_local_tutor.conv1.weight).item())
#                     print("Norm of 1st layer of tutee:  ", torch.norm(net_local_tutee.conv1.weight).item())

#             if ii==0:
#                 total_params_tutor = "{:,}".format(sum(p.numel() for p in net_local_tutor.parameters())) 
#                 total_params_tutee = "{:,}".format(sum(p.numel() for p in net_local_tutee.parameters()))
#                 print("Parameters of tutor:         ", total_params_tutor)
#                 print("Parameters of tutee:         ", total_params_tutee)

#             ##  TUTOR training  #####################################################################################################################
#             ### head training will failure learning rate's effects ##################
#             # weight_tutor, loss_tutor, std_tutor = local_tutor.train(net=net_local_tutor.to(args.device), body_lr=lr, head_lr=lr) # dj: private tutor (here we use full network update)
#             weight_tutor, loss_tutor = local_tutor.train(net=net_local_tutor.to(args.device), body_lr=lr, head_lr=0.) # dj: private tutor (here we use full network update)
#             # directly store the private tutor's weight per clinet in this epoch #######################
#             update_keys_tutor = list(weight_tutor.keys())
#             weight_tutor = {k: v for k, v in weight_tutor.items() if k in update_keys_tutor}   ###### dj: private tutor (suppose full network update, body + head )
#             net_local_tutor_list[idx].load_state_dict(weight_tutor, strict=False) ###### dj: private tutor
#             ############################################################################################            
#             acc_test_tutor, loss_test_tutor = test_img_local(net_local_tutor, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])
            

#             ##  TUTEE training without KD  ##########################################################################################################
#             # update_layer = update_layers[ep] # DJ
#             ## TODO
#             # each client use different lr !!!!!!!!!!!!!!!!!!!!!!!!!!!
#             # class LocalUpdate(object) @ models/Update.py 
#             # if args.local_upt_part == 'body':
#             weight_tutee_kd, loss_tutee_kd = local_tutee.train(net=net_local_tutee.to(args.device), body_lr=lr, head_lr=0.)
#             # if args.local_upt_part == 'head':
#             #     weight_tutee, loss_tutee, std_tutee = local_tutee.train(net=net_local_tutee.to(args.device), body_lr=0., head_lr=lr)
#             # if args.local_upt_part == 'full':
#             #     weight_tutee, loss_tutee, std_tutee = local_tutee.train(net=net_local_tutee.to(args.device), body_lr=lr, head_lr=lr)
#             acc_test_tutee, loss_test_tutee = test_img_local(net_local_tutee, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx]) 


#             if acc_test_tutor>acc_test_tutee:
#                 net_local_tutee = copy.deepcopy(net_local_tutee_list[idx])
#                 ##  TUTEE training with KD  #############################################################################################################
#                 # the tutor should be trained first
#                 ### head training will failure learning rate's effects ##################
#                 # if args.local_upt_part == 'body':
#                 weight_tutee_kd, loss_tutee_kd = local_tutee.train_knowledge_distillation(trainedTutor=net_local_tutor.to(args.device), tutee=net_local_tutee.to(args.device), body_lr=lr, head_lr=0., Temporature=args.temperature, KD_weight=args.KD_weight)
#                 # if args.local_upt_part == 'head':
#                 #     weight_tutee_kd, loss_tutee_kd = local_tutee.train_knowledge_distillation(trainedTutor=net_local_tutor[idx].to(args.device), tutee=net_local_tutee.to(args.device), body_lr=0., head_lr=lr, Temporature=args.temperature, KD_weight=args.KD_weight)
#                 # if args.local_upt_part == 'full':
#                 # weight_tutee_kd, loss_tutee_kd = local_tutee.train_knowledge_distillation(trainedTutor=net_local_tutor[idx].to(args.device), tutee=net_local_tutee.to(args.device), body_lr=lr, head_lr=lr, Temporature=args.temperature, KD_weight=args.KD_weight)
#                 ############################################################################################
#             weight_tutee = weight_tutee_kd
#             loss_tutee   = loss_tutee_kd
#             std_tutee    = 0.0


#             # tutee only for communication
#             weight_local = copy.deepcopy(weight_tutee) 
#             loss_local_pool.append(copy.deepcopy(loss_tutee))
#             std_local_pool.append(copy.deepcopy(std_tutee))



#             # update client loss history
#             loss_clients[idx] = loss_tutee



#             # TODO
#             # the guy of better test accuracy is to be the teacher #################################
#             if True:
#                 # acc_test_tutor, loss_test_tutor = test_img_local(net_local_tutor, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])
#                 # acc_test_tutee, loss_test_tutee = test_img_local(net_local_tutee, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx]) 
#                 tutor_acc_pool.append('{:>5.1f}'.format(acc_test_tutor))           
#                 tutee_acc_pool.append('{:>5.1f}'.format(acc_test_tutee))
                                






#             ################################################################################
#             # simulate each client (idx) upload its model weights (update_list) w.r.t. specific layer defined in (layer_updater)
#             ################################################################################
#             update_list = []
#             for m in range(num_module):
#                 if idx in layer_updater[m]:                                                         # clients to update some specific modules 
#                     update_list += module_pool[m]                                                   # which layer-weights this client has to update 
#                     layer_updater_idx[m].append(ii)                                                 # which layers this client has to update 
#             # print(idx, update_list)
#             # each client only update its layer-weights defined in update_list
#             # name_list = [name for name, _ in net_local_tutee.named_parameters()]
#             # weight_list = [weight for name, weight in net_local_tutee_list[idx].named_parameters()]     # contain old weights
#             for name in tutee_layer_name_list:  # tutee_layer_name_list are trainable paras not including nontrainable paras
#                 if name in update_list:
#                     layer_update_cnt_per_epoch[name]  += 1
#                     layer_update_cnt_all_epochs[name] += 1
#                 else:    
#                     # weight_local[name] *= (0.0*torch.ones_like(weight_local[name]).to('cuda:2')).type(torch.long)
#                     weight_local[name] = 0.
#             ################################################################################
    
#             # each client use the same weight for aggregation
#             if uniform_aggregation:   
#                 if weight_global is None:
#                     weight_global  = copy.deepcopy(weight_local)
#                 else:
#                     # with torch.no_grad():
#                     for k in weight_global.keys():
#                         weight_global[k] += weight_local[k]                             
#             # each client use different weight for aggregation
#             else:
#                 # with torch.no_grad():     
#                 weight_local_pool.append(copy.deepcopy(weight_local)) # DJ: stote all local weights for selection

#         if True:
#             print('TUTOR test acc in this epoch:', tutor_acc_pool)
#             print('TUTEE test acc in this epoch:', tutee_acc_pool)
            

#         # remove duplicate index
#         # breakpoint()
#         for m in range(num_module):
#             layer_updater_idx[m] = list(set(layer_updater_idx[m])) # per-layer indices of the selected clients
#         # breakpoint()
#         toc_local_update = datetime.now()
#         print('Local update time: %.3f seconds.' % ((toc_local_update-tic_local_update).total_seconds()))




#         # Aggregation ----------------------------------------------------------------------------------------------------------------
#         # print(layer_update_cnt_per_epoch)
#         # print(layer_update_cnt_all_epochs)
#         if uniform_aggregation:
#             for k in weight_global.keys():
#                 if k in tutee_layer_name_list:
#                     weight_global[k] = torch.div(weight_global[k], layer_update_cnt_per_epoch[k]) # DJ: trainable parameters
#                 else:
#                     weight_global[k] = torch.div(weight_global[k], num_users_per_epoch)       # DJ: parameters not trainable
#             # loss
#             loss_avg = sum(loss_local_pool) / len(idxs_users)  # uniform loss weights
#         else:
#             importance_all = softmax(std_local_pool)                 # importance per client
#             # breakpoint()
#             importance_layer_updater = []
#             if num_updaters_per_module!=0:
#                 # importance_FBer = softmax(std_local_pool[:num_updaters_per_module]) # importance per client a module
#                 for m in range(num_module):
#                     importance_layer_updater.append(softmax([std_local_pool[i] for i in layer_updater_idx[m]])) # importance per client in each module

#             for i_local in range(len(idxs_users)):
#                 selected_local_weights = weight_local_pool[i_local]        # get the slected local weights
#                 selected_local_index   = idxs_users[i_local]

#                 if i_local == 0:
#                     weight_global = copy.deepcopy(selected_local_weights)    # just get a weight name dictionary, ignore their weights
#                     for k in weight_global.keys():
#                         weight_global[k] = 0.0 * selected_local_weights[k]
                
#                 if num_updaters_per_module==len(idxs_users):                     # all clients are full body update
#                     for k in weight_global.keys():
#                         weight_global[k] += importance_all[i_local]*selected_local_weights[k]
#                     continue

#                 for m in range(num_module):                          # client contributes to multiple modules
#                     if i_local in layer_updater_idx[m]:  
#                         for k in module_pool[m]:
#                             weight_global[k] += importance_layer_updater[m][layer_updater_idx[m].index(i_local)]*selected_local_weights[k] 
#             # loss
#             loss_avg = sum(softmax(std_local_pool)*loss_local_pool) # DJ: softmax loss weight

#         # FedBABU+ (classifier update in the server)
#         if args.server_data_ratio > 0.0:
#             server = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train['server'])
#             net_tutee.load_state_dict(weight_global, strict=True)
#             weight_global, loss = server.train(net=net_tutee.to(args.device), body_lr=lr, head_lr=0., local_eps=int(args.results_save[-1]))
        
#         # Broadcast ------------------------------------------------------------------------------------------------------------------
#         update_keys = list(weight_global.keys())
#         if args.aggr_part == 'body':
#             if args.server_data_ratio > 0.0:
#                 pass
#             else:
#                 update_keys = [k for k in update_keys if 'linear' not in k]
#         elif args.aggr_part == 'head':
#             update_keys = [k for k in update_keys if 'linear' in k]
#         elif args.aggr_part == 'full':
#             pass
#         weight_global = {k: v for k, v in weight_global.items() if k in update_keys}
#         for user_idx in range(args.num_users): # DJ: all clients 100
#             net_local_tutee_list[user_idx].load_state_dict(weight_global, strict=False)

#         print('learning rate: ', lr)
#         if (ep + 1) in lr_schedule:
#             lr *= 0.1

#         # Evaluating -----------------------------------------------------------------------------------------------------------------
#         args.test_freq = 10
#         if (ep + 1) % args.test_freq == 0:
#             tic_eval = datetime.now() # DJ
#             acc_test_tutor, loss_test_tutor = test_img_local_all(net_local_tutor_list, args, dataset_test, dict_users_test, return_all=False)
#             acc_test_tutee, loss_test_tutee = test_img_local_all(net_local_tutee_list, args, dataset_test, dict_users_test, return_all=False)
#             acc_test_tutor_all, loss_test_tutor_all = test_img_local_all(net_local_tutor_list, args, dataset_test, dict_users_test, return_all=True)
#             acc_test_tutee_all, loss_test_tutee_all = test_img_local_all(net_local_tutee_list, args, dataset_test, dict_users_test, return_all=True)
#             print('ROUND {:3d}: Test loss (Tutor/Tutee) {:.3f}/{:.3f}, Test accuracy (Tutor/Tutee) {:>5.2f}/{:>5.2f}'.format(ep+1, loss_test_tutor, loss_test_tutee, acc_test_tutor, acc_test_tutee))
#             print(acc_test_tutor_all)
#             print(acc_test_tutee_all)
#             if best_acc is None or acc_test_tutee > best_acc:
#                 net_best = copy.deepcopy(net_tutee)
#                 best_acc = acc_test_tutee
#                 best_epoch = ep
#                 best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
#                 torch.save(net_local_tutee_list[0].state_dict(), best_save_path)
                
# #                 for user_idx in range(args.num_users):
# #                     best_save_path = os.path.join(base_dir, algo_dir, 'best_local_{}.pt'.format(user_idx))
# #                     torch.save(net_local_tutee_list[user_idx].state_dict(), best_save_path)

#             results.append(np.array([ep, loss_avg, loss_test_tutee, acc_test_tutee, best_acc]))
#             final_results = np.array(results)
#             final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test_tutee', 'acc_test_tutee', 'best_acc'])
#             final_results.to_csv(results_save_path, index=False)
#             toc_eval = datetime.now()
#             print('Evaluation time: %.3f seconds.' % ((toc_eval-tic_eval).total_seconds()))
#         swriter.add_scalar('ACC-Round', acc_test_tutee, ep+1)  # DJ    
#         swriter.add_scalar('ACC-Round-Tutee', acc_test_tutee, ep+1)  # DJ
#         swriter.add_scalar('ACC-Round-Tutor', acc_test_tutor, ep+1)  # DJ
#         swriter.add_scalar('Loss-Round', loss_avg, ep+1) # DJ 
#     swriter.close()      # DJ   
#     toc = datetime.now() # DJ
    
#     print('Best model, epoch: {}, acc: {}'.format(best_epoch, best_acc))
#     print('Elapsed time: %.3f minutes.' % ((toc-tic).total_seconds()/60))    # DJ 



















































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