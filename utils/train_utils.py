from torchvision import datasets, transforms
from models.Nets import CNNCifar, MobileNetCifar, MobileNetCifarTiny
from models.ResNet import ResNet18, ResNet34, ResNet50, ResNet18Tiny
from utils.sampling import iid, noniid, iid_unbalanced, noniid_unbalanced
from prettytable import PrettyTable # JC
import random                       # JC
import math

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def get_data(args, env='fed'):
    if env == 'single':
        if args.dataset == 'cifar10':
            dataset_train = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=trans_cifar10_train)
            dataset_test = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=trans_cifar10_val)
            
        elif args.dataset == 'cifar100':
            dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar100_train)
            dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        return dataset_train, dataset_test
    
    elif env == 'fed':
        if args.unbalanced:
            if args.dataset == 'cifar10':
                dataset_train = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=trans_cifar10_train)
                dataset_test = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=trans_cifar10_val)
                if args.iid:
                    dict_users_train = iid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size)
                    dict_users_test = iid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size)
                else:
                    dict_users_train, rand_set_all = noniid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user)
                    dict_users_test, rand_set_all = noniid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar100':
                dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar100_train)
                dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar100_val)
                if args.iid:
                    dict_users_train = iid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size)
                    dict_users_test = iid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size)
                else:
                    dict_users_train, rand_set_all = noniid_unbalanced(dataset_train, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user)
                    dict_users_test, rand_set_all = noniid_unbalanced(dataset_test, args.num_users, args.num_batch_users, args.moved_data_size, args.shard_per_user, rand_set_all=rand_set_all)
            else:
                exit('Error: unrecognized dataset')

        else:
            if args.dataset == 'mnist':
                dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
                dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
                # sample users
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar10':
                dataset_train = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=trans_cifar10_train)
                dataset_test = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=trans_cifar10_val)
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            elif args.dataset == 'cifar100':
                dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar100_train)
                dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar100_val)
                if args.iid:
                    dict_users_train = iid(dataset_train, args.num_users, args.server_data_ratio)
                    dict_users_test = iid(dataset_test, args.num_users, args.server_data_ratio)
                else:
                    # breakpoint()
                    dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio)
                    dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio, rand_set_all=rand_set_all)
            else:
                exit('Error: unrecognized dataset')

        return dataset_train, dataset_test, dict_users_train, dict_users_test

def get_model(args):
    # dj: add net_tutor, net_glob > net_tutee
    if args.model == 'cnn' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'mobile' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = MobileNetCifar(num_classes=args.num_classes).to(args.device)
    elif args.model == 'mobiletiny' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = MobileNetCifarTiny(num_classes=args.num_classes).to(args.device)        
    elif args.model == 'resnet18' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = ResNet18(num_classes=args.num_classes).to(args.device)  
    elif args.model == 'resnet18tiny' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = ResNet18Tiny(num_classes=args.num_classes).to(args.device)                
    elif args.model == 'mobileKD' and args.dataset in ['cifar10', 'cifar100']:
        net_tutor = MobileNetCifar(num_classes=args.num_classes).to(args.device)
        net_tutee = MobileNetCifarTiny(num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet18KD' and args.dataset in ['cifar10', 'cifar100']:
        net_tutor = ResNet18(num_classes=args.num_classes).to(args.device)
        net_tutee = ResNet18Tiny(num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet34' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = ResNet34(num_classes=args.num_classes).to(args.device)        
    elif args.model == 'resnet50' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = ResNet50(num_classes=args.num_classes).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp' and args.dataset == 'mnist':
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    if args.model == 'mobileKD' or args.model == 'resnet18KD':
        return net_tutor, net_tutee
    else:
        return net_glob

def get_layer_list(model):                        # DJ
    layer_list = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        layer_list.append(name)
    return layer_list


def count_model_parameters(model):                        # DJ
    table = PrettyTable(["Modules", "Parameters", "AccumParas"])
    total_params = 0
    accum_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        accum_params+=params
        table.add_row([name, params, accum_params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def count_layer_parameters(model, layers):                        # DJ
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        if name in layers:
            params = parameter.numel()
            total_params+=params
    return total_params

def update_scheme(epochs=320, scheme="E", warmup=0, lr_schedule=[0]):         # DJ
    update_layers=[]
    if warmup != 0 and warmup == int(warmup):
        update_layers += [999]*int(warmup)

    # Random
    if scheme == "R":
        print("Use Random update scheme")
        for iter in range(epochs):
            update_layers.append(random.randint(0, 3))
    # Entire
    elif scheme == "E":
        print("Use Entire update scheme")
        for iter in range(epochs):
            update_layers.append(999)

    elif scheme == "Test0":
        print("Use Customize-Reverse update scheme")
        order = [0,0,0,0] # case 0
        for i in range(epochs//len(order)):
            for j in range(len(order)):
                update_layers.append(order[j%len(order)])            
    elif scheme == "Test1":
        print("Use Customize-Reverse update scheme")
        order = [1,1,1,1] # case 1
        for i in range(epochs//len(order)):
            for j in range(len(order)):
                update_layers.append(order[j%len(order)])  
    elif scheme == "Test2":
        print("Use Customize-Reverse update scheme")
        order = [2,2,2,2] # case 2
        for i in range(epochs//len(order)):
            for j in range(len(order)):
                update_layers.append(order[j%len(order)])  
    elif scheme == "Test3":
        print("Use Customize-Reverse update scheme")
        order = [3,3,3,3] # case 3
        for i in range(epochs//len(order)):
            for j in range(len(order)):
                update_layers.append(order[j%len(order)])    
    elif scheme == "Test4":
        print("Use Customize-Reverse update scheme")
        order = [0,0,1,1,2,2,3,3] # case 4
        for i in range(epochs//len(order)):
            for j in range(len(order)):
                update_layers.append(order[j%len(order)]) 
    elif scheme == "Test5":
        print("Use Customize-Reverse update scheme")
        order = [0,1,2,3,0,1,2,3] # case 5
        for i in range(epochs//len(order)):
            for j in range(len(order)):
                update_layers.append(order[j%len(order)]) 
    elif scheme == "Test6":
        print("Use Customize-Reverse update scheme")
        order = [0,1,2,3,999,999,999,999] # case 6
        for i in range(epochs//len(order)):
            for j in range(len(order)):
                update_layers.append(order[j%len(order)]) 
    elif scheme == "Test7":
        print("Use Piece-Wise update scheme")
        remained_epochs = epochs-int(warmup)
        order = [0,1,2,3] # case 7
        for i in range(len(order)):
            for j in range(remained_epochs // len(order)):
                update_layers.append(order[i%len(order)])
        if len(update_layers) < epochs:
            for k in range(100):
                update_layers.append(order[-1])            
    elif scheme == "Test8":
        print("Use Piece-Wise update scheme")
        remained_epochs = epochs-int(warmup)
        order = [1,2,3] # case 8
        for i in range(len(order)):
            for j in range(remained_epochs // len(order)):
                update_layers.append(order[i%len(order)])
        if len(update_layers) < epochs:
            for k in range(100):
                update_layers.append(order[-1])   
    elif scheme == "Test9":
        print("Use Piece-Wise update scheme")
        order = [999,0,1,2,3] # case 9
        lr_schedule.append(epochs) # consider the last epochs split
        for i in range(len(lr_schedule)):
            # breakpoint()
            if lr_schedule[i] > warmup:
                if i==0:
                    remained_epochs = lr_schedule[i] - max(warmup, 0)
                else:
                    remained_epochs = lr_schedule[i] - max(warmup, lr_schedule[i-1])
                for j in range(len(order)):
                    for k in range(remained_epochs // len(order)):
                        update_layers.append(order[j%len(order)])
        if len(update_layers) < epochs:
            for i in range(100):
                update_layers.append(order[-1])      
    elif scheme == "Test10":
        print("Use Piece-Wise update scheme")
        order = [999,1,2,3] # case 10
        lr_schedule.append(epochs) # consider the last epochs split
        for i in range(len(lr_schedule)):
            # breakpoint()
            if lr_schedule[i] > warmup:
                if i==0:
                    remained_epochs = lr_schedule[i] - max(warmup, 0)
                else:
                    remained_epochs = lr_schedule[i] - max(warmup, lr_schedule[i-1])
                for j in range(len(order)):
                    for k in range(remained_epochs // len(order)):
                        update_layers.append(order[j%len(order)])
        if len(update_layers) < epochs:
            for i in range(100):
                update_layers.append(order[-1])     
    elif scheme == "Test11":
        print("Use Piece-Wise update scheme")
        # warmup is become a ratio between two learning rate changings
        order = [0,1,2,3] # case 11
        lr_schedule.append(epochs) # consider the last epochs split
        for i in range(len(lr_schedule)):
            # breakpoint()
            if i==0:
                num_epochs = lr_schedule[i]
            else:
                num_epochs = lr_schedule[i] - lr_schedule[i-1]
            warmup_epochs = math.floor(num_epochs*warmup)
            update_layers += [999]*int(warmup_epochs)
            remained_epochs = num_epochs - warmup_epochs
            for j in range(len(order)):
                for k in range(remained_epochs // len(order)):
                    update_layers.append(order[j%len(order)])
        if len(update_layers) < epochs:
            print("need padding")
            for i in range(100):
                update_layers.append(order[-1])     
    elif scheme == "Test12":
        print("Use Piece-Wise update scheme")
        # warmup is become a ratio between two learning rate changings
        order = [1,2,3] # case 12
        lr_schedule.append(epochs) # consider the last epochs split
        for i in range(len(lr_schedule)):
            # breakpoint()
            if i==0:
                num_epochs = lr_schedule[i]
            else:
                num_epochs = lr_schedule[i] - lr_schedule[i-1]
            warmup_epochs = math.floor(num_epochs*warmup)
            update_layers += [999]*int(warmup_epochs)
            remained_epochs = num_epochs - warmup_epochs
            for j in range(len(order)):
                for k in range(remained_epochs // len(order)):
                    update_layers.append(order[j%len(order)])
        if len(update_layers) < epochs:
            print("need padding")
            for i in range(100):
                update_layers.append(order[-1])     
    elif scheme == "Test13":
        print("Use Piece-Wise update scheme")
        # warmup is become a ratio between two learning rate changings
        order = [0,1,2,3] # case 13
        lr_schedule.append(epochs) # consider the last epochs split
        for i in range(len(lr_schedule)):
            # breakpoint()
            if i==0:
                num_epochs = lr_schedule[i]
            else:
                num_epochs = lr_schedule[i] - lr_schedule[i-1]
            warmup_epochs = math.floor(num_epochs*warmup)
            warmup_post_ratio = 0.5
            update_layers += [999]*int(warmup_epochs*(1-warmup_post_ratio))
            remained_epochs = num_epochs - warmup_epochs
            for j in range(len(order)):
                for k in range(remained_epochs // len(order)):
                    update_layers.append(order[j%len(order)])
            update_layers += [999]*int(lr_schedule[i] - len(update_layers))
        if len(update_layers) < epochs:
            print("need padding >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for i in range(100):
                update_layers.append(order[-1])
    elif scheme == "Test14":
        print("Use Piece-Wise update scheme")
        # warmup is become a ratio between two learning rate changings
        order = [0,1,2] # case 14
        lr_schedule.append(epochs) # consider the last epochs split
        for i in range(len(lr_schedule)):
            # breakpoint()
            if i==0:
                num_epochs = lr_schedule[i]
            else:
                num_epochs = lr_schedule[i] - lr_schedule[i-1]
            warmup_epochs = math.floor(num_epochs*warmup)
            warmup_post_ratio = 0.5
            update_layers += [999]*int(warmup_epochs*(1-warmup_post_ratio))
            remained_epochs = num_epochs - warmup_epochs
            for j in range(len(order)):
                for k in range(remained_epochs // len(order)):
                    update_layers.append(order[j%len(order)])
            update_layers += [999]*int(lr_schedule[i] - len(update_layers))
        if len(update_layers) < epochs:
            print("need padding >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for i in range(100):
                update_layers.append(order[-1])                                                                       
    elif scheme == "Test15":
        print("Use Piece-Wise update scheme")
        # warmup is become a ratio between two learning rate changings
        order = [0,1,2,3] # case 15
        lr_schedule.append(epochs) # consider the last epochs split
        for i in range(len(lr_schedule)):
            # breakpoint()
            if i==0:
                num_epochs = lr_schedule[i]
            else:
                num_epochs = lr_schedule[i] - lr_schedule[i-1]
            warmup_epochs = math.floor(num_epochs*warmup)
            warmup_post_ratio = 0.25
            update_layers += [999]*int(warmup_epochs*(1-warmup_post_ratio))
            remained_epochs = num_epochs - warmup_epochs
            for j in range(len(order)):
                for k in range(remained_epochs // len(order)):
                    update_layers.append(order[j%len(order)])
            update_layers += [999]*int(lr_schedule[i] - len(update_layers))
        if len(update_layers) < epochs:
            print("need padding >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for i in range(100):
                update_layers.append(order[-1])
    elif scheme == "Test16":
        print("Use Piece-Wise update scheme")
        # warmup is become a ratio between two learning rate changings
        order = [0,1,2] # case 16
        lr_schedule.append(epochs) # consider the last epochs split
        for i in range(len(lr_schedule)):
            # breakpoint()
            if i==0:
                num_epochs = lr_schedule[i]
            else:
                num_epochs = lr_schedule[i] - lr_schedule[i-1]
            warmup_epochs = math.floor(num_epochs*warmup)
            warmup_post_ratio = 0.25
            update_layers += [999]*int(warmup_epochs*(1-warmup_post_ratio))
            remained_epochs = num_epochs - warmup_epochs
            for j in range(len(order)):
                for k in range(remained_epochs // len(order)):
                    update_layers.append(order[j%len(order)])
            update_layers += [999]*int(lr_schedule[i] - len(update_layers))
        if len(update_layers) < epochs:
            print("need padding >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for i in range(100):
                update_layers.append(order[-1])                                                                                       
    else:
        exit('wrong update scheme')

    return update_layers[:epochs]
