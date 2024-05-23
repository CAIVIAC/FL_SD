'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=True)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes, track_running_stats=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=True)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18Cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet18Cifar, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear_01 = nn.Linear( 64*block.expansion, num_classes, bias=True)
        self.linear_02 = nn.Linear( 64*block.expansion, num_classes, bias=True)
        self.linear_03 = nn.Linear( 64*block.expansion, num_classes, bias=True)
        self.linear_1 = nn.Linear( 64*block.expansion, num_classes, bias=True)
        self.linear_2 = nn.Linear(128*block.expansion, num_classes, bias=True)
        self.linear_3 = nn.Linear(128*block.expansion, num_classes, bias=True)
        # self.linear_4 = nn.Linear(256*block.expansion, num_classes+1, bias=True)
        self.linear_5 = nn.Linear(256*block.expansion, num_classes, bias=True)
        # self.linear_6 = nn.Linear(512*block.expansion, num_classes+1, bias=True)
        self.linear_f  = nn.Linear(512*block.expansion, num_classes,   bias=True)
        self.linear_f0 = nn.Linear(512*block.expansion, num_classes,   bias=True)
        self.linear_f1 = nn.Linear(512*block.expansion, num_classes,   bias=True)
        self.linear_f2 = nn.Linear(512*block.expansion, num_classes,   bias=True)
        self.linear_alpha01 = nn.Linear( 64*block.expansion, 1, bias=True)
        self.linear_alpha02 = nn.Linear( 64*block.expansion, 1, bias=True)
        self.linear_alpha03 = nn.Linear( 64*block.expansion, 1, bias=True)
        self.linear_alpha1 = nn.Linear( 64*block.expansion, 1, bias=True)
        self.linear_alpha2 = nn.Linear(128*block.expansion, 1, bias=True)
        self.linear_alpha3 = nn.Linear(128*block.expansion, 1, bias=True)
        self.linear_alpha5 = nn.Linear(256*block.expansion, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_layer=None, stop_scale=True):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        if out_layer==None:
            out_layer = -1
        
        # define network list ------------------------------------------------
        outs = []                                                                        # DJ: store all intermediate features       
        for l in range(len(self.layer1)):
            out = self.layer1[l](out)
            outs.append(out)    
        for l in range(len(self.layer2)):
            out = self.layer2[l](out)
            outs.append(out)
        for l in range(len(self.layer3)):
            out = self.layer3[l](out)
            outs.append(out)
        for l in range(len(self.layer4)):
            out = self.layer4[l](out)
            outs.append(out)

        # define FINAL logits of complete network
        outs_this = outs[-1]
        outs_this = F.avg_pool2d(outs_this, outs_this.shape[2])
        outs_this = outs_this.view(outs_this.size(0), -1)                       
        logits_final = self.linear_f(outs_this)    #*self.alphaf

        # define AUXILIARY logits ------------------------------------------  
        logit_list = []


        outs_0 = outs[0]
        outs_0 = F.avg_pool2d(outs_0, outs_0.shape[2])
        outs_0 = outs_0.view(outs_0.size(0), -1)        #*self.alpha2               
        logits_0 = self.linear_01(outs_0)                #*self.alpha2
        if stop_scale:
            logits_0 =  logits_0
        else:
            # logits_1 =  logits_1[:,:100] / (self.sigmoid(logits_1[:,-1])[:,None]      + 1e-1)
            logits_0 =  logits_0[:,:100] / (self.sigmoid(self.linear_alpha01(outs_0))      + 1e-1)
            # logits_1 =  logits_1 * self.linear_alpha1(outs_1)
        logit_list.append(logits_0)

 
        outs_0 = outs[0]
        outs_0 = F.avg_pool2d(outs_0, outs_0.shape[2])
        outs_0 = outs_0.view(outs_0.size(0), -1)        #*self.alpha2               
        logits_0 = self.linear_02(outs_0)                #*self.alpha2
        if stop_scale:
            logits_0 =  logits_0
        else:
            # logits_1 =  logits_1[:,:100] / (self.sigmoid(logits_1[:,-1])[:,None]      + 1e-1)
            logits_0 =  logits_0[:,:100] / (self.sigmoid(self.linear_alpha02(outs_0))      + 1e-2)
            # logits_1 =  logits_1 * self.linear_alpha1(outs_1)
        logit_list.append(logits_0)

        outs_0 = outs[0]
        outs_0 = F.avg_pool2d(outs_0, outs_0.shape[2])
        outs_0 = outs_0.view(outs_0.size(0), -1)        #*self.alpha2               
        logits_0 = self.linear_03(outs_0)                #*self.alpha2
        if stop_scale:
            logits_0 =  logits_0
        else:
            # logits_1 =  logits_1[:,:100] / (self.sigmoid(logits_1[:,-1])[:,None]      + 1e-1)
            logits_0 =  logits_0[:,:100] / (self.sigmoid(self.linear_alpha03(outs_0))      + 1e-3)
            # logits_1 =  logits_1 * self.linear_alpha1(outs_1)
        logit_list.append(logits_0)        








        # outs_1 = outs[1]
        # outs_1 = F.avg_pool2d(outs_1, outs_1.shape[2])
        # outs_1 = outs_1.view(outs_1.size(0), -1)        #*self.alpha2               
        # logits_1 = self.linear_1(outs_1)                #*self.alpha2
        # if stop_scale:
        #     logits_1 =  logits_1[:,:100]
        # else:
        #     # logits_1 =  logits_1[:,:100] / (self.sigmoid(logits_1[:,-1])[:,None]      + 1e-1)
        #     # breakpoint()
        #     logits_1 =  logits_1[:,:100] / (self.sigmoid(self.linear_alpha1(outs_1))      + 1e-1)
        #     # logits_1 =  logits_1 * self.linear_alpha1(outs_1)
        # logit_list.append(logits_1) 

        # outs_2 = outs[2]
        # outs_2 = F.avg_pool2d(outs_2, outs_2.shape[2])
        # outs_2 = outs_2.view(outs_2.size(0), -1)        #*self.alpha2               
        # logits_2 = self.linear_2(outs_2)                #*self.alpha2
        # if stop_scale:
        #     logits_2 =  logits_2[:,:100]
        # else:
        #     logits_2 =  logits_2[:,:100] / (self.sigmoid(self.linear_alpha2(outs_2))      + 1e-1)
        # logit_list.append(logits_2) 



        # outs_3 = outs[3]
        # outs_3 = F.avg_pool2d(outs_3, outs_3.shape[2])
        # outs_3 = outs_3.view(outs_3.size(0), -1)        #*self.alpha3               
        # logits_3 = self.linear_3(outs_3)                #*self.alpha3
        # if stop_scale:
        #     logits_3 =  logits_3[:,:100]
        # else:
        #     # logits_3 =  logits_3[:,:100] / (self.sigmoid(logits_3[:,-1])[:,None]      + 1e-1) 
        #     logits_3 =  logits_3[:,:100] / (self.sigmoid(self.linear_alpha3(outs_3))      + 1e-1)
        #     # logits_3 =  logits_3 * self.linear_alpha3(outs_3)
        # logit_list.append(logits_3) 

        # outs_5 = outs[5]
        # outs_5 = F.avg_pool2d(outs_5, outs_5.shape[2])
        # outs_5 = outs_5.view(outs_5.size(0), -1)        #*self.alpha5               
        # logits_5 = self.linear_5(outs_5)                #*self.alpha5
        # if stop_scale:
        #     logits_5 =  logits_5[:,:100]
        # else:
        #     # logits_5 =  logits_5[:,:100] / (self.sigmoid(logits_5[:,-1])[:,None]      + 1e-1)    
        #     logits_5 =  logits_5[:,:100] / (self.sigmoid(self.linear_alpha5(outs_5))      + 1e-1)  
        #     # logits_5 =  logits_5 * self.linear_alpha5(outs_5)
        # logit_list.append(logits_5) 

    

    
        # outs_6 = outs[6]
        # outs_6 = F.avg_pool2d(outs_6, outs_6.shape[2])
        # outs_6 = outs_6.view(outs_6.size(0), -1)        #*self.alpha6               
        # logits_6 = self.linear_6(outs_6)                #*self.alpha6
        # logits_6 =  logits_6[:,:100] * logits_6[:,-1][:,None]
        # logit_list.append(logits_6) 

        if out_layer == -1:
            return logits_final
        else:
            return logits_final, logit_list
        


class ResNet50Cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet50Cifar, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # dj: bias = False
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.linear_2  = nn.Linear( 64*block.expansion, num_classes, bias=True)       # dj: bias = True
        self.linear_6  = nn.Linear(128*block.expansion, num_classes, bias=True)
        self.linear_12 = nn.Linear(256*block.expansion, num_classes, bias=True)
        self.linear_f  = nn.Linear(512*block.expansion, num_classes, bias=True)

        self.linear_alpha2  = nn.Linear( 64*block.expansion, 1, bias=True) # dj: not to learn with above linear layer is better
        self.linear_alpha6  = nn.Linear(128*block.expansion, 1, bias=True)
        self.linear_alpha12 = nn.Linear(256*block.expansion, 1, bias=True)
        self.linear_alphaf  = nn.Linear(512*block.expansion, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_layer=None, stop_scale=True):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        if out_layer==None:
            out_layer = -1
        
        # define network list ------------------------------------------------
        outs = []                                                                        # DJ: store all intermediate features       
        for l in range(len(self.layer1)):
            out = self.layer1[l](out)
            outs.append(out)    
        for l in range(len(self.layer2)):
            out = self.layer2[l](out)
            outs.append(out)
        for l in range(len(self.layer3)):
            out = self.layer3[l](out)
            outs.append(out)
        for l in range(len(self.layer4)):
            out = self.layer4[l](out)
            outs.append(out)

        # define FINAL logits of complete network
        outs_this = outs[-1]
        outs_this = F.avg_pool2d(outs_this, outs_this.shape[2])
        outs_this = outs_this.view(outs_this.size(0), -1)                       
        logits_final = self.linear_f(outs_this)    #*self.alphaf
        if stop_scale:
            logits_final =  logits_final
            # scale = self.sigmoid(self.linear_alphaf(outs_this))
            # logits_final =  logits_final / (scale      *2)
        else:
            logits_final =  logits_final / (self.sigmoid(self.linear_alphaf(outs_this))      + 0.5)
            # scale = self.sigmoid(self.linear_alphaf(outs_this))
            # logits_final =  logits_final / (scale.clamp(min=0.1, max=1.0))
        if torch.isnan(logits_final).any():
            print("NAN!    Prediction: ", self.linear_alphaf(outs_this), "Sigmoid: ", self.sigmoid(self.linear_alphaf(outs_this)))
            logits_final =  logits_final


        # define AUXILIARY logits ------------------------------------------  
        logit_list = []
        outs_2 = outs[2]
        outs_2 = F.avg_pool2d(outs_2, outs_2.shape[2])
        outs_2 = outs_2.view(outs_2.size(0), -1)        #*self.alpha2               
        logits_2 = self.linear_2(outs_2)                #*self.alpha2
        if stop_scale:
            logits_2 =  logits_2
            # scale = self.sigmoid(self.linear_alpha2(outs_2))
            # logits_2 =  logits_2 / (scale      *2)
        else:
            logits_2 =  logits_2 / (self.sigmoid(self.linear_alpha2(outs_2))      + 0.5)
            # scale = self.sigmoid(self.linear_alpha2(outs_2))
            # logits_2 =  logits_2 / (scale.clamp(min=0.1, max=1.0))
        if torch.isnan(logits_2).any():
            print("NAN!    Prediction: ", self.linear_alpha2(outs_2), "Sigmoid: ", self.sigmoid(self.linear_alpha2(outs_2)))  
            logits_2 =  logits_2         
        logit_list.append(logits_2) 
 
    
        outs_6 = outs[6]
        outs_6 = F.avg_pool2d(outs_6, outs_6.shape[2])
        outs_6 = outs_6.view(outs_6.size(0), -1)        #*self.alpha2               
        logits_6 = self.linear_6(outs_6)                #*self.alpha2
        if stop_scale:
            logits_6 =  logits_6
            # scale = self.sigmoid(self.linear_alpha6(outs_6))
            # logits_6 =  logits_6 / (scale      *2)
        else:
            logits_6 =  logits_6 / (self.sigmoid(self.linear_alpha6(outs_6))      + 0.5)
            # scale = self.sigmoid(self.linear_alpha6(outs_6))
            # logits_6 =  logits_6 / (scale.clamp(min=0.1, max=1.0))
        if torch.isnan(logits_6).any():
            print("NAN!    Prediction: ", self.linear_alpha6(outs_6), "Sigmoid: ", self.sigmoid(self.linear_alpha6(outs_6)))  
            logits_6 =  logits_6            
        logit_list.append(logits_6) 

        outs_12 = outs[12]
        outs_12 = F.avg_pool2d(outs_12, outs_12.shape[2])
        outs_12 = outs_12.view(outs_12.size(0), -1)        #*self.alpha2               
        logits_12 = self.linear_12(outs_12)                #*self.alpha2
        if stop_scale:
            logits_12 =  logits_12
            # scale = self.sigmoid(self.linear_alpha12(outs_12))
            # logits_12 =  logits_12 / (scale      *2)
        else:
            logits_12 =  logits_12 / (self.sigmoid(self.linear_alpha12(outs_12))      + 0.5)
            # scale = self.sigmoid(self.linear_alpha12(outs_12))
            # logits_12 =  logits_12 / (scale.clamp(min=0.1, max=1.0))
        if torch.isnan(logits_12).any():
            print("NAN!    Prediction: ", self.linear_alpha12(outs_12), "Sigmoid: ", self.sigmoid(self.linear_alpha12(outs_12)))   
            logits_12 =  logits_12        
        logit_list.append(logits_12) 


        if out_layer == -1:
            return logits_final
        else:
            return logits_final, logit_list


def ResNet18(num_classes=10):
    return ResNet18Cifar(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)     # paras: 11,220,132


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    # breakpoint()
    return ResNet50Cifar(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)









class ResNetTiny(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetTiny, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # mask it by ResNet(BasicBlock, [2, 2]
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # mask it by ResNet(BasicBlock, [2, 2, 2]
        # self.linear = nn.Linear(512*block.expansion, num_classes)           # mask it by ResNet(BasicBlock, [2, 2, 2]
        # self.linear = nn.Linear(256, num_classes)                           # add it by ResNet(BasicBlock, [2, 2, 2]
        self.linear = nn.Linear(128, num_classes)                             # add it by ResNet(BasicBlock, [2, 2]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.layer3(out)           # mask it by ResNet(BasicBlock, [2, 2]
        # out = self.layer4(out)           # mask it by ResNet(BasicBlock, [2, 2, 2]
        # breakpoint()
        # out = F.avg_pool2d(out, 4)       # mask it by ResNet(BasicBlock, [2, 2, 2]
        # out = F.avg_pool2d(out, 8)         # add it by ResNet(BasicBlock, [2, 2, 2]
        out = F.avg_pool2d(out, 16)         # add it by ResNet(BasicBlock, [2, 2]
        out = out.view(out.size(0), -1)
        # breakpoint()
        out = self.linear(out)
        return out

 
def ResNet18Tiny(num_classes=10):
    # return ResNet(BasicBlock, [2, 2, 1, 1], num_classes=num_classes)   # paras:  5,318,820
    # return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)   # paras:  4,949,412
    # return ResNet(BasicBlock, [2, 2, 2], num_classes=num_classes)      # paras:  2,800,804
    # return ResNet(BasicBlock, [2, 2], num_classes=num_classes)         # paras:    688,292
    # return ResNet(BasicBlock, [2, 2, 1], num_classes=num_classes)      # paras:  1,620,132
    return ResNetTiny(BasicBlock, [2, 3], num_classes=num_classes)       # paras:    983,716

