#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import pdb
import copy
from torch.optim import Optimizer
from datetime import datetime # DJ


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.loss_func_per = nn.CrossEntropyLoss(reduction='none')
        self.selected_clients = []
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        # dj: num_workers=4 to faster the data loading
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=4, pin_memory=True)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr, out_layer=-1, local_eps=None):
        net.train()

        # train and update
        
        # For ablation study
        """
        body_params = []
        head_params = []
        for name, p in net.named_parameters():
            if 'features.0' in name or 'features.1' in name: # active
                body_params.append(p)
            else: # deactive
                head_params.append(p)
        """
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]

        # head_names = [name for name, p in net.named_parameters() if 'linear' in name]
        # print(head_names)
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],  
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # tic = datetime.now()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                # breakpoint()
                # logits = net(images, out_layer)
                logits = net(images)
                # breakpoint()

                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                # toc = datetime.now()
                # print(batch_idx, idx, 'One batch time: %.3f seconds.' % ((toc-tic).total_seconds()))

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    

    # Multi-Heads Knowledge Distillation
    def train_MHKD(self, net, body_lr, head_lr, out_layer=-1, local_eps=None, Temporature=1, KD_weight = 1005):
        net.train()
        # For ablation study
        """
        body_params = []
        head_params = []
        for name, p in net.named_parameters():
            if 'features.0' in name or 'features.1' in name: # active
                body_params.append(p)
            else: # deactive
                head_params.append(p)
        """
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]

        # breakpoint()
        # head_names = [name for name, p in net.named_parameters() if 'linear' in name]
        # print(head_names)
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            # print(iter)
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # tic = datetime.now()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                # case 1: only tutor   44.5, 1.0
                if KD_weight==1:
                    logits_tutor = net(images, -1, True)                        # last output as tutor
                    loss = self.loss_func(logits_tutor, labels)
                    net.zero_grad()
                    loss.backward()
                    optimizer.step()

                # case 2: only student   23.4, 1.2
                if KD_weight==2:
                    logits_tutee, logits_list = net(images, 0, True) # intermediate output as tutee
                    loss = self.loss_func(logits_list[0], labels)
                    net.zero_grad()
                    loss.backward()
                    optimizer.step()           

                # case 3: only student   23.4, 1.2
                if KD_weight==3:
                    logits_tutee, logits_list = net(images, 1, True) # intermediate output as tutee
                    loss = self.loss_func(logits_list[1], labels)
                    net.zero_grad()
                    loss.backward()
                    optimizer.step() 

                # case 4: only student   23.4, 1.2
                if KD_weight==4:
                    logits_tutee, logits_list = net(images, 2, True) # intermediate output as tutee
                    loss = self.loss_func(logits_list[2], labels)
                    net.zero_grad()
                    loss.backward()
                    optimizer.step()                     

                # # case 3: share weights, simutaneous training: 44.1, 14.6
                # if KD_weight==3:
                #     logits_tutor = net(images, -1)                        # last output as tutor
                #     logits_tutee, dummy1, dummy2 = net(images, out_layer) # intermediate output as tutee
                #     loss = self.loss_func(logits_tutor, labels) + self.loss_func(logits_tutee, labels)
                #     net.zero_grad()
                #     loss.backward()
                #     optimizer.step()

                # # case 4: share weights, sequential training: 46.0, 17.0
                # if KD_weight==4:
                #     logits_tutor = net(images, -1)                        # last output as tutor
                #     loss = self.loss_func(logits_tutor, labels)
                #     net.zero_grad()
                #     loss.backward()
                #     optimizer.step()
                #     logits_tutee, dummy1, dummy2 = net(images, out_layer) # intermediate output as tutee
                #     loss = self.loss_func(logits_tutee, labels)
                #     net.zero_grad()
                #     loss.backward()
                #     optimizer.step()

                # # case 5: share weights, sequential training: 44.5, 18.0
                # if KD_weight==5:
                #     logits_tutee, dummy1 = net(images, out_layer) # intermediate output as tutee
                #     loss = self.loss_func(logits_tutee, labels)
                #     net.zero_grad()
                #     loss.backward()
                #     optimizer.step() 
                #     logits_tutor = net(images, -1)                        # last output as tutor
                #     loss = self.loss_func(logits_tutor, labels)
                #     net.zero_grad()
                #     loss.backward()
                #     optimizer.step()
                   




 
#    soft_prob = nn.functional.log_softmax(tutee_logits / Temporature, dim=-1)
                # dont care if there exists any correct prediction
                if KD_weight==1002:  # no correct label vs. correct label : 57.2 40.7 vs. 57.7 43.0
                    logits_final, logits_aux = net(images, out_layer, False)    # intermediate output as tutee
                    loss_aux, loss_kd = 0., 0.
                    for l in range(len(logits_aux)):
                        loss_aux += self.loss_func(logits_aux[l], labels)
                    for l in range(len(logits_aux)):
                        loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[l] / Temporature, dim=-1), nn.functional.log_softmax(logits_final / Temporature, dim=-1)) * (Temporature**2)
                    loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*10 + loss_kd*1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                # if there exists any correct final prediction, then one way KD with all predictions
                if KD_weight==1003: # no correct label vs. correct label : 57.2 40.7 vs. 57.7 43.0
                    logits_final, logits_aux = net(images, out_layer, False)    # intermediate output as tutee
                    y_pred = logits_final.data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                    KD_Idx = y_pred.eq(labels.data)  # only take the correct index
                    loss_aux, loss_kd = 0., 0.
                    for l in range(len(logits_aux)):
                        loss_aux += self.loss_func(logits_aux[l], labels)
                    if any(KD_Idx):
                        for l in range(len(logits_aux)):
                            loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[l] / Temporature, dim=-1), nn.functional.log_softmax(logits_final / Temporature, dim=-1)) * (Temporature**2)
                    # else:
                        # print('    exist no correct prediction from last layer!')
                    loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*10 + loss_kd*1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # if there exists any correct final prediction, then one way KD with correct predictions
                if KD_weight==1004: # no correct label vs. correct label : 57.2 40.7 vs. 57.7 43.0
                    logits_final, logits_aux = net(images, out_layer, False)    # intermediate output as tutee
                    y_pred = logits_final.data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                    KD_Idx = y_pred.eq(labels.data)  # only take the correct index
                    loss_aux, loss_kd = 0., 0.
                    for l in range(len(logits_aux)):
                        loss_aux += self.loss_func(logits_aux[l], labels)
                    if any(KD_Idx):
                        for l in range(len(logits_aux)):
                            loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[l][KD_Idx] / Temporature, dim=-1), nn.functional.log_softmax(logits_final[KD_Idx] / Temporature, dim=-1)) * (Temporature**2) 
                    # else:
                        # print('    exist no correct prediction from last layer!')
                    loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*10 + loss_kd*1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # if there exists any correct bidirectional final prediction, then one way KD with bidirectional correct predictions
                if KD_weight==1005: # no correct label vs. correct label : 57.2 40.7 vs. 57.7 43.0
                    # print('model 1005')
                    # logits_final, logits_aux = net(images, out_layer, True)    # intermediate output as tutee
                    # logits_final, logits_aux = net(images, out_layer, True)    # for resnet18
                    logits_final, logits_aux = net(images, out_layer, False)    # for mobileNet, 4convNet
                    y_pred = logits_final.data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                    KD_Idx = y_pred.eq(labels.data)  # only take the correct index
                    loss_aux, loss_kd = 0., 0.
                    for l in range(len(logits_aux)):
                        loss_aux += self.loss_func(logits_aux[l], labels)
                        # print(logits_aux[l])
                    if any(KD_Idx):
                        for l in range(len(logits_aux)):
                            loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[l][KD_Idx] / Temporature, dim=-1), nn.functional.log_softmax(logits_final[KD_Idx] / Temporature, dim=-1)) * (Temporature**2)
                            y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                            KD_Idx_reverse = y_pred_reverse.eq(labels.data)  # only take the correct index
                            if any(KD_Idx_reverse):
                                loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
                    else:
                        # print('    exist no correct prediction from last layer!')
                        for l in range(len(logits_aux)):
                            y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                            KD_Idx_reverse = y_pred_reverse.eq(labels.data)  # only take the correct index
                            if any(KD_Idx_reverse):
                                loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
                    # loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*0.01 + loss_kd*0.1
                    # loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*0.01 + loss_kd*0.1 # for resnet18
                    # loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*10 + loss_kd*1 # for mobileNet
                    loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*0.01 + loss_kd*1 # for 4convNet
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()



                if KD_weight==0: # no correct label vs. correct label : 57.2 40.7 vs. 57.7 43.0
                    # print('model 1005')
                    logits_final, logits_aux = net(images, out_layer, True)    # for resnet50
                    loss = self.loss_func(logits_final, labels) * 1 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # if there exists any correct bidirectional final prediction, then one way KD with bidirectional correct predictions
                if KD_weight==10050: # no correct label vs. correct label : 57.2 40.7 vs. 57.7 43.0
                    # print('model 1005')
                    logits_final, logits_aux = net(images, out_layer, True)    # for resnet50
                    # logits_final, logits_aux = net(images, out_layer, True)    # for resnet18
                    # logits_final, logits_aux = net(images, out_layer, False)    # for mobileNet
                    y_pred = logits_final.data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                    KD_Idx = y_pred.eq(labels.data)  # only take the correct index
                    loss_aux, loss_kd = 0., 0.
                    for l in range(len(logits_aux)):
                        loss_aux += self.loss_func(logits_aux[l], labels)
                        # print(logits_aux[l])
                    if any(KD_Idx):
                        for l in range(len(logits_aux)):
                            loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[l][KD_Idx] / Temporature, dim=-1), nn.functional.log_softmax(logits_final[KD_Idx] / Temporature, dim=-1)) * (Temporature**2)
                            y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                            KD_Idx_reverse = y_pred_reverse.eq(labels.data)  # only take the correct index
                            if any(KD_Idx_reverse):
                                loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
                    else:
                        # print('    exist no correct prediction from last layer!')
                        for l in range(len(logits_aux)):
                            y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                            KD_Idx_reverse = y_pred_reverse.eq(labels.data)  # only take the correct index
                            if any(KD_Idx_reverse):
                                loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
                    loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*1 + loss_kd*1     # for resnet50
                    # loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*0.01 + loss_kd*0.1 # for resnet18
                    # loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*10 + loss_kd*1     # for mobileNet
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if KD_weight==10051: # no correct label vs. correct label : 57.2 40.7 vs. 57.7 43.0
                    # print('model 1005')
                    logits_final, logits_aux = net(images, out_layer, False)    # for resnet50
                    # logits_final, logits_aux = net(images, out_layer, True)    # for resnet18
                    # logits_final, logits_aux = net(images, out_layer, False)    # for mobileNet
                    y_pred = logits_final.data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                    KD_Idx = y_pred.eq(labels.data)  # only take the correct index
                    loss_aux, loss_kd = 0., 0.
                    for l in range(len(logits_aux)):
                        loss_aux += self.loss_func(logits_aux[l], labels)
                        # print(logits_aux[l])
                    if any(KD_Idx):
                        for l in range(len(logits_aux)):
                            loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[l][KD_Idx] / Temporature, dim=-1), nn.functional.log_softmax(logits_final[KD_Idx] / Temporature, dim=-1)) * (Temporature**2)
                            y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                            KD_Idx_reverse = y_pred_reverse.eq(labels.data)  # only take the correct index
                            if any(KD_Idx_reverse):
                                loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
                    else:
                        # print('    exist no correct prediction from last layer!')
                        for l in range(len(logits_aux)):
                            y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                            KD_Idx_reverse = y_pred_reverse.eq(labels.data)  # only take the correct index
                            if any(KD_Idx_reverse):
                                loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
                    loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*1 + loss_kd*1     # for resnet50
                    # loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*0.01 + loss_kd*0.1 # for resnet18
                    # loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*10 + loss_kd*1     # for mobileNet
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 

                # if there exists any correct bidirectional final prediction, then one way KD with bidirectional correct predictions
                if KD_weight==1001: # no correct label vs. correct label : 57.2 40.7 vs. 57.7 43.0
                    # print('model 1005')
                    logits_final, logits_aux = net(images, out_layer, True)    # intermediate output as tutee
                    y_pred = logits_final.data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                    KD_Idx = y_pred.eq(labels.data)  # only take the correct index
                    loss_aux, loss_kd = 0., 0.
                    for l in range(len(logits_aux)):
                        loss_aux += self.loss_func(logits_aux[l], labels)
                    if any(KD_Idx):
                        for l in range(len(logits_aux)):
                            loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[l][KD_Idx] / Temporature, dim=-1), nn.functional.log_softmax(logits_final[KD_Idx] / Temporature, dim=-1)) * (Temporature**2)
                            y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                            KD_Idx_reverse = y_pred_reverse.eq(labels.data)  # only take the correct index
                            if any(KD_Idx_reverse):
                                loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
                    else:
                        # print('    exist no correct prediction from last layer!')
                        for l in range(len(logits_aux)):
                            y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                            KD_Idx_reverse = y_pred_reverse.eq(labels.data)  # only take the correct index
                            if any(KD_Idx_reverse):
                                loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
                    loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*10 + loss_kd*1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 

                if KD_weight==100512: # for finetune
                    logits_final, logits_aux = net(images, out_layer, False)    # intermediate output as tutee
                    y_pred = logits_final.data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                    KD_Idx = y_pred.eq(labels.data)  # only take the correct index
                    loss_aux = 0.0
                    loss_kd =  0.0
                    for l in range(len(logits_aux)):
                        loss_aux += self.loss_func(logits_aux[l], labels)
                    if any(KD_Idx):
                        for l in range(len(logits_aux)):
                            loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_aux[l][KD_Idx] / Temporature, dim=-1), nn.functional.log_softmax(logits_final[KD_Idx] / Temporature, dim=-1)) * (Temporature**2)
                            y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                            KD_Idx_reverse = y_pred_reverse.eq(labels.data)  # only take the correct index
                            if any(KD_Idx_reverse):
                                loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
                    else:
                        # print('    exist no correct prediction from last layer!')
                        for l in range(len(logits_aux)):
                            y_pred_reverse = logits_aux[l].data.max(1, keepdim=True)[1].view_as(labels)  # get the index of the max log-probability
                            KD_Idx_reverse = y_pred_reverse.eq(labels.data)  # only take the correct index
                            if any(KD_Idx_reverse):
                                loss_kd  += nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_final[KD_Idx_reverse] / Temporature, dim=-1), nn.functional.log_softmax(logits_aux[l][KD_Idx_reverse] / Temporature, dim=-1)) * (Temporature**2) 
                    loss = self.loss_func(logits_final, labels) * 1 +  loss_aux*0 + loss_kd*1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()







                if KD_weight==12: # BACKUP ############################# T = 2, kdloss * 1, 52.2, 44.6
                    logits_tutor = net(images, -1)           # last output as tutor
                    loss = self.loss_func(logits_tutor, labels) * 1
                    # optimizer.zero_grad()                               # 50.6  41.8
                    loss.backward() 
                    optimizer.step()
                    logits_tutee, dummy1, dummy2 = net(images, out_layer)    # intermediate output as tutee 
                    with torch.no_grad():                    # Forward pass with the tutor model - do not save gradients here as we do not change the tutor's weights
                        logits_tutor = net(images, -1)  
                    kd_loss = nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_tutee), nn.functional.log_softmax(logits_tutor)) * (Temporature**2)
                    loss = kd_loss * 50 + self.loss_func(logits_tutee, labels) * 10       # kdloss 10:  52.3 44.8 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                #  + self.loss_func(logits_add, labels) * 10
                if KD_weight==101:
                    #  1 10 10: 56 35  or this (prefer tutor)
                    #  1  1 10: 51 40  this    (care tutee)
                    #  1  0 10: 50 42
                    #  1  0  0: 44  1
                    #  1  1  1: 51 13
                    # 10  1  1: 50  5
                    #  1 10  1: 57  4  or this (prefer tutor)
                    # 10 10 10: 54 25
                    # https://stats.stackexchange.com/questions/346299/whats-the-effect-of-scaling-a-loss-function-in-deep-learning
                    # Without regularization, using SGD optimizer: scaling loss by α is equivalent to scaling SGD's learning rate by α.  (That is this!!)
                    # Without regularization, using Nadam: scaling loss by α has no effect.
                    # With regularization, using either SGD or Nadam optimizer: changing the scale of prediction loss will affect the trade-off between prediction loss and regularization.
                    logits_tutee, logits_tutor, logits_aux = net(images, out_layer)    # intermediate output as tutee
                    kd_loss = nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_tutee), nn.functional.log_softmax(logits_tutor).detach()) * (Temporature**2) 
                    # loss = self.loss_func(logits_tutee, labels) * 10 + self.loss_func(logits_tutor, labels) * 1 + self.loss_func(logits_aux, labels) * 1 + kd_loss * 1
                    loss =  self.loss_func(logits_tutor, labels) * 1 + kd_loss * 1 + self.loss_func(logits_tutee, labels) * 10
                    # model = net.to(self.args.device)
                    # w1 = model.conv1.weight.detach().clone() # w1 = body_params[30].grad
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # model_updated = net.to(self.args.device)
                    # w2 = model_updated.conv1.weight.detach().clone() # w2 = body_params[30].grad
                    # print(torch.mean(abs(w2 - w1)))

                if KD_weight==104: # TL method: two heads (better than 102, 103, 105)
                    logits_tutee, logits_tutor, logits_aux = net(images, out_layer)
                    loss = self.loss_func(logits_tutor, labels) * 1  + self.loss_func(logits_tutee, labels) * 1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()   
                    logits_tutee, logits_tutor, dummy = net(images, out_layer, out_layer+1)    # intermediate output as tutee 
                    kd_loss = nn.KLDivLoss(log_target=True)(nn.functional.log_softmax(logits_tutee), nn.functional.log_softmax(logits_tutor))
                    loss = kd_loss * 1 + self.loss_func(logits_tutee, labels) * 10   
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()



 
       

                batch_loss.append(loss.item())
                # toc = datetime.now()
                # print(batch_idx, idx, 'One batch time: %.3f seconds.' % ((toc-tic).total_seconds()))

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)



    # def train(self, net, body_lr, head_lr, local_eps=None):
    #     net.train()

    #     # train and update
        
    #     # For ablation study
    #     """
    #     body_params = []
    #     head_params = []
    #     for name, p in net.named_parameters():
    #         if 'features.0' in name or 'features.1' in name: # active
    #             body_params.append(p)
    #         else: # deactive
    #             head_params.append(p)
    #     """
    #     body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
    #     head_params = [p for name, p in net.named_parameters() if 'linear' in name]
    #     optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
    #                                  {'params': head_params, 'lr': head_lr}],
    #                                 momentum=self.args.momentum,
    #                                 weight_decay=self.args.wd)

 
    #     # init_params   = [p for name, p in net.named_parameters() if 'layer' not in name and 'linear' not in name]
    #     # layer1_params = [p for name, p in net.named_parameters() if 'layer1' in name]
    #     # layer2_params = [p for name, p in net.named_parameters() if 'layer2' in name]
    #     # layer3_params = [p for name, p in net.named_parameters() if 'layer3' in name]
    #     # layer4_params = [p for name, p in net.named_parameters() if 'layer4' in name]
    #     # head_params = [p for name, p in net.named_parameters() if 'linear' in name]
    #     # optimizer = torch.optim.SGD([{'params': init_params,   'lr': body_lr},
    #     #                              {'params': layer1_params, 'lr': body_lr},
    #     #                              {'params': layer2_params, 'lr': body_lr},
    #     #                              {'params': layer3_params, 'lr': body_lr},
    #     #                              {'params': layer4_params, 'lr': head_lr},
    #     #                              {'params': head_params,   'lr': head_lr}],
    #     #                             momentum=self.args.momentum,
    #     #                             weight_decay=self.args.wd)

    #     epoch_loss = []
    #     tic_here = datetime.now()
    #     if local_eps is None:
    #         if self.pretrain:
    #             local_eps = self.args.local_ep_pretrain
    #         else:
    #             local_eps = self.args.local_ep
    #     # toc_here = datetime.now()  
    #     # print('Elapsed time: %.3f seconds.' % ((toc_here-tic_here).total_seconds()))        
    #     for iter in range(local_eps):
    #         batch_loss = []
    #         tic_here = datetime.now()
    #         for batch_idx, (images, labels) in enumerate(self.ldr_train):
                
    #             images, labels = images.to(self.args.device), labels.to(self.args.device)
 
    #             logits = net(images)
               


    #             # loss_all = self.loss_func(logits, labels) # dj: line 32 -> line 33 (get loss per training sample)
    #             # loss = torch.mean(loss_all)               # dj: since line 32 -> line 33
    #             loss = self.loss_func(logits, labels)


    #             # tic_here = datetime.now()
    #             optimizer.zero_grad()
    #             # toc_here = datetime.now()
    #             # print(batch_idx, '     Elapsed time: %.3f seconds.' % ((toc_here-tic_here).total_seconds()))

    #             # tic_here = datetime.now()
    #             loss.backward()
    #             # toc_here = datetime.now()
    #             # print(batch_idx, '        Elapsed time: %.3f seconds.' % ((toc_here-tic_here).total_seconds()))
                
    #             # tic_here = datetime.now()
    #             optimizer.step()
    #             # toc_here = datetime.now()
    #             # print(batch_idx, '            Elapsed time: %.3f seconds.' % ((toc_here-tic_here).total_seconds()))

                
    #             batch_loss.append(loss.item())
    #             toc_here = datetime.now()
                
           
    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))
    #         # print(batch_idx, '            Elapsed time: %.3f seconds.' % ((toc_here-tic_here).total_seconds()))
    #         # breakpoint()
        
    #     return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # def train(self, net, body_lr, head_lr, local_eps=None):
    #     net.train()
    #     g_net = copy.deepcopy(net) # for fedprox #########################
 
    #     body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
    #     head_params = [p for name, p in net.named_parameters() if 'linear' in name]
    #     paras_dict = [{'params': body_params, 'lr': body_lr},{'params': head_params, 'lr': head_lr}]
 
    #     optimizer = torch.optim.SGD(paras_dict,
    #                                 momentum=self.args.momentum,
    #                                 weight_decay=self.args.wd)

    #     epoch_loss = []

    #     if local_eps is None:
    #         if self.pretrain:
    #             local_eps = self.args.local_ep_pretrain
    #         else:
    #             local_eps = self.args.local_ep
    #     # dj: each epoch: one client trained with 500 images
    #     for iter in range(local_eps):
    #         batch_loss = []
    #         # dj: images: 50x3x32x32, labels: 50, logits: 50x100
    #         # dj: each batch: one client trained with 50 images
    #         labels_samples = [] # dj
    #         loss_samples = [] # dj
    #         for batch_idx, (images, labels) in enumerate(self.ldr_train):
    #             images, labels = images.to(self.args.device), labels.to(self.args.device)
    #             net.zero_grad()
    #             logits = net(images)

    #             loss_all = self.loss_func(logits, labels) # dj: line 32 -> line 33 (get loss per training sample)
    #             loss = torch.mean(loss_all)               # dj: since line 32 -> line 33
    #             # for fedprox ######################### 
    #             # fed_prox_reg = 0.0
    #             # for l_param, g_param in zip(net.parameters(), g_net.parameters()):   # dj this regularization operation slower the training
    #             #     fed_prox_reg += (self.args.mu / 2 * torch.norm((l_param - g_param)) ** 2)
    #             # loss += fed_prox_reg
    #             #######################################
    #             loss.backward()
    #             optimizer.step()

    #             batch_loss.append(loss.item())
    #         ##############################################################################
    #             labels_samples.append(labels.tolist()) # dj
    #             loss_samples.append(loss_all.tolist()) # dj
    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))

    #         labels_samples = [item for sublist in labels_samples for item in sublist] # dj
    #         loss_samples = [item for sublist in loss_samples for item in sublist]     # dj
    #         import numpy as np
    #         from scipy.stats import iqr  # interquartile range (IQR)
    #         from scipy import stats      # median absolute deviation (MAD)
    #         labels_samples = np.array(labels_samples)
    #         loss_samples = np.array(loss_samples)
    #         uniques = np.unique(labels_samples)
    #         std_per_class = []
    #         for i in range(len(uniques)):
    #             loss_list_per_unique = loss_samples[labels_samples == uniques[i]]
    #             # std_per_class.append(np.std(loss_list_per_unique))
    #             std_per_class.append(iqr(loss_list_per_unique, rng=(40,60)))
    #             # std_per_class.append(stats.median_abs_deviation(loss_list_per_unique))
    #         mean_std = np.mean(std_per_class)
    #         negative_mean_std = -mean_std
    #         # print(uniques)
    #         # print(std_per_class)
    #         # print(mean_std)
    #         ##############################################################################
    #     return net.state_dict(), sum(epoch_loss) / len(epoch_loss), negative_mean_std
    
    # knowledge_distillation
    def train_KD(self, trainedTutor, tutee, body_lr, head_lr, out_layer=-1, local_eps=None, Temporature=5.0, KD_weight = 10):
        
        trainedTutor.eval()  # tutor set to evaluation mode
        tutee.train()        # tutee set to train mode

        body_params = [p for name, p in tutee.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in tutee.named_parameters() if 'linear' in name]
        paras_dict = [{'params': body_params, 'lr': body_lr},{'params': head_params, 'lr': head_lr}]
        optimizer = torch.optim.SGD(paras_dict, momentum=self.args.momentum, weight_decay=self.args.wd)

        epoch_loss = []
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        # dj: each epoch: one client trained with 500 images
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # Forward pass with the tutor model - do not save gradients here as we do not change the tutor's weights
                with torch.no_grad():
                    tutor_logits = trainedTutor(images, -1)

                # Forward pass with the tutee model
                tutee_logits = tutee(images, out_layer)

                # #Soften the tutee logits by applying softmax first and log() second
                # soft_targets = nn.functional.softmax(tutor_logits / Temporature, dim=-1)
                # soft_prob = nn.functional.log_softmax(tutee_logits / Temporature, dim=-1)

                # # # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                # soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] #* (Temporature**2)

                # # it seems that temporature < 1 should not sclaed by T**2
                # kd_loss = nn.KLDivLoss()(soft_prob, soft_targets) #* (Temporature**2)


                # # Calculate the true label loss
                tutee_loss = self.ce_loss(tutee_logits, labels)
                # # print('tutee_loss, soft_targets_loss2: ',tutee_loss, kd_loss)

                # tutor_loss = self.ce_loss(tutor_logits, labels)



                # prediction loss per image in a batch
                # kd_loss_each = torch.mean(nn.KLDivLoss(reduce=False)(soft_prob, soft_targets),dim=1)
                # tutor_loss_each = self.loss_func(tutor_logits, labels)
                # tutee_loss_each = self.loss_func(tutee_logits, labels)
                # IG_Idx = tutor_loss_each  > tutee_loss_each # tutee has a better prediction
                # KD_Idx = tutor_loss_each <= tutee_loss_each # tutor ha a better prediction
                # scale_weight = len(images)/torch.sum(KD_Idx) # less samples has higher weight
                # kd_loss = torch.mean(kd_loss_each[KD_Idx]) #* scale_weight
                # # breakpoint()


                # Weighted sum of the two losses
                if KD_weight == 0:
                    loss = tutee_loss * 1
                elif KD_weight == 100:
                    loss = tutee_loss * 1 + kd_loss * 10
                    # if tutor_loss>tutee_loss:
                    #     loss = tutee_loss * 10
                elif KD_weight == 101:
                    loss = tutee_loss * 10 + kd_loss * 10
                    # if tutor_loss>tutee_loss:
                    #     loss = tutee_loss * 10
                elif KD_weight == 102:    
                    loss = (tutee_loss * 1 + kd_loss * 10) * 10
                    # if tutor_loss>tutee_loss:    # 49.0
                    #     loss = tutee_loss * 10   
                elif KD_weight == 103:
                    loss = tutee_loss * 10
                else:
                    Temporature = 2
                    soft_targets = nn.functional.softmax(tutor_logits / Temporature, dim=-1)
                    soft_prob = nn.functional.log_softmax(tutee_logits / Temporature, dim=-1)
                    soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (Temporature**2)
                    loss = 0.25*soft_targets_loss + 0.75*tutee_loss
                # loss = tutee_loss * 1                                       # 33.44             (mobileNetTiny            , KD_weight=10) 
                                                                              # 47.78             (resnet18Tiny             , KD_weight=10, tutor=45.74)
                    
                # loss = tutee_loss * 1 + kd_loss * KD_weight                # 32.95/32.39       (mobileNetTiny/focal      , KD_weight=10) 
                                                                              # 47.57/48.43/47.65 (resnet18Tiny/KD_Idx/focal, KD_weight=10, tutor=45.74)
                                                                              # 47.97/47.97/41.11 (resnet18Tiny ?/KO/FO, tutor = 50.52)
                # loss = tutee_loss * KD_weight + kd_loss * KD_weight        # 42.37/43.18       (mobileNetTiny/focal      , KD_weight=10) 
                                                                              # 49.16/49.26/49.42 (resnet18Tiny/KD_Idx/focal, KD_weight=10, tutor=45.74)
                                                                              # 49.68/49.98/48.78 (resnet18Tiny ?/KO/FO, tutor = 50.52)
                # loss = (tutee_loss * 1 + kd_loss * KD_weight) * KD_weight  # 42.03/42.09       (mobileNetTiny/focal      , KD_weight=10) 
                                                                              # 49.32/49.47/49.80 (resnet18Tiny/KD_Idx/focal, KD_weight=10, tutor=45.74)
                                                                              # 49.43/49.66/42.05 (resnet18Tiny ?/KO/FO, tutor = 50.52)
                # loss = tutee_loss * KD_weight                               # 42.11             (mobileNetTiny            , KD_weight=10)             
                                                                              # 49.36             (resnet18Tiny             , KD_weight=10, tutor=45.74)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return tutee.state_dict(), sum(epoch_loss) / len(epoch_loss)














class LocalUpdatePerFedAvg(object):    
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, lr, beta=0.001, momentum=0.9):
        net.train()
        # train and update

        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        
        epoch_loss = []
        
        for local_ep in range(self.args.local_ep):
            batch_loss = []
            
            if len(self.ldr_train) / self.args.local_ep == 0:
                num_iter = int(len(self.ldr_train) / self.args.local_ep)
            else:
                num_iter = int(len(self.ldr_train) / self.args.local_ep) + 1
                
            train_loader_iter = iter(self.ldr_train)
            
            for batch_idx in range(num_iter):
                temp_net = copy.deepcopy(list(net.parameters()))
                    
                # Step 1
                for g in optimizer.param_groups:
                    g['lr'] = lr
                    
                try:
                    images, labels = next(train_loader_iter)
                except:
                    train_loader_iter = iter(self.ldr_train)
                    images, labels = next(train_loader_iter)
                    
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                net.zero_grad()
                
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                
                
                # Step 2
                for g in optimizer.param_groups:
                    g['lr'] = beta
                    
                try:
                    images, labels = next(train_loader_iter)
                except:
                    train_loader_iter = iter(self.ldr_train)
                    images, labels = next(train_loader_iter)
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                    
                net.zero_grad()
                
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                
                # restore the model parameters to the one before first update
                for old_p, new_p in zip(net.parameters(), temp_net):
                    old_p.data = new_p.data.clone()
                    
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) 
    
    def one_sgd_step(self, net, lr, beta=0.001, momentum=0.9):
        net.train()
        # train and update

        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        
        test_loader_iter = iter(self.ldr_train)

        # Step 1
        for g in optimizer.param_groups:
            g['lr'] = lr

        try:
            images, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(self.ldr_train)
            images, labels = next(train_loader_iter)


        images, labels = images.to(self.args.device), labels.to(self.args.device)

        net.zero_grad()

        logits = net(images)

        loss = self.loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        # Step 2
        for g in optimizer.param_groups:
            g['lr'] = beta

        try:
            images, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(self.ldr_train)
            images, labels = next(train_loader_iter)

        images, labels = images.to(self.args.device), labels.to(self.args.device)

        net.zero_grad()

        logits = net(images)

        loss = self.loss_func(logits, labels)
        loss.backward()

        optimizer.step()


        return net.state_dict()

class LocalUpdateFedRep(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, lr):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': 0.0, 'name': "body"},
                                     {'params': head_params, 'lr': lr, "name": "head"}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        local_eps = self.args.local_ep
        
        for iter in range(local_eps):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
        
        for g in optimizer.param_groups:
            if g['name'] == "body":
                g['lr'] = lr
            elif g['name'] == 'head':
                g['lr'] = 0.0
        
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            logits = net(images)

            loss = self.loss_func(logits, labels)
            loss.backward()
            optimizer.step()

        return net.state_dict()

class LocalUpdateFedProx(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr):
        net.train()
        g_net = copy.deepcopy(net)
        
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                
                # for fedprox
                fed_prox_reg = 0.0
                for l_param, g_param in zip(net.parameters(), g_net.parameters()):
                    fed_prox_reg += (self.args.mu / 2 * torch.norm((l_param - g_param)) ** 2)
                loss += fed_prox_reg
                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) 
    
class LocalUpdateDitto(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
            
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, w_ditto=None, lam=0, idx=-1, lr=0.1, last=False, momentum=0.9):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
                
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

        local_eps = self.args.local_ep
        args = self.args 
        epoch_loss=[]
        num_updates = 0
        
        for iter in range(local_eps):
            done=False
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                w_0 = copy.deepcopy(net.state_dict())
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if w_ditto is not None:
                    w_net = copy.deepcopy(net.state_dict())
                    for key in w_net.keys():
                        w_net[key] = w_net[key] - args.lr*lam*(w_0[key] - w_ditto[key])
                    net.load_state_dict(w_net)
                    optimizer.zero_grad()
                
                num_updates += 1
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    
