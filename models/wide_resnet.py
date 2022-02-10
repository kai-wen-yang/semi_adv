from __future__ import print_function
import abc
import os
import math

import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
import sys


class MySequential(nn.Sequential):
    def forward(self, x, adv):
        for module in self._modules.values():
            x = module(x, adv=adv)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(wide_basic, self).__init__()
        self.bn_adv_flag = bn_adv_flag
        self.bn_adv_momentum = bn_adv_momentum
        self.bn1 = nn.BatchNorm2d(in_planes)
        if self.bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(in_planes,  momentum=self.bn_adv_momentum)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.bn_adv_flag:
            self.bn2_adv = nn.BatchNorm2d(planes, momentum=self.bn_adv_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x, adv=False):
        if adv and self.bn_adv_flag:
            out = self.dropout(self.conv1(F.relu(self.bn1_adv(x))))
            out = self.conv2(F.relu(self.bn2_adv(out)))
        else:
            out = self.dropout(self.conv1(F.relu(self.bn1(x))))
            out = self.conv2(F.relu(self.bn2(out)))

        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, bn_adv_flag=False, bn_adv_momentum=0.01):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        self.bn_adv_flag = bn_adv_flag
        self.bn_adv_momentum = bn_adv_momentum

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, bn_adv_flag=self.bn_adv_flag,
                                       bn_adv_momentum=bn_adv_momentum)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        if bn_adv_flag:
            self.bn1_adv = nn.BatchNorm2d(nStages[3], momentum=self.bn_adv_momentum)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, bn_adv_flag=False, bn_adv_momentum=0.1):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(
                block(self.in_planes, planes, dropout_rate, stride, bn_adv_flag=bn_adv_flag, bn_adv_momentum=bn_adv_momentum))
            self.in_planes = planes

        return MySequential(*layers)

    def forward(self, x, return_feature=False, adv=False):
        out = self.conv1(x)
        out = self.layer1(out, adv)
        out = self.layer2(out, adv)
        out = self.layer3(out, adv)
        if adv and self.bn_adv_flag:
            out = F.relu(self.bn1_adv(out))
        else:
            out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if return_feature:
            return out, feature
        else:
            return out