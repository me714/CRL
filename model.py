# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :CeNet
# @File     :model
# @Date     :2022/3/7 20:27
# @Author   :Sun
# @Email    :szqqishi@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import nni.retiarii.nn.pytorch

from torchtools import *
from collections import OrderedDict
import math
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#
#
# class ConvBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, userelu=True, momentum=0.1, affine=True, track_running_stats=True):
#         super(ConvBlock, self).__init__()
#         self.layers = nn.Sequential()
#         self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
#                                                  kernel_size=3, stride=1, padding=1, bias=False))
#
#         if tt.arg.normtype == 'batch':
#             self.layers.add_module('Norm', nn.BatchNorm2d(out_planes, momentum=momentum, affine=affine,
#                                                           track_running_stats=track_running_stats))
#         elif tt.arg.normtype == 'instance':
#             self.layers.add_module('Norm', nn.InstanceNorm2d(out_planes))
#
#         if userelu:
#             self.layers.add_module('ReLU', nn.ReLU(inplace=True))
#
#         self.layers.add_module(
#             'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
#
#     def forward(self, x):
#         out = self.layers(x)
#         return out
#
#
# class ConvNet(nn.Module):
#     def __init__(self, opt, momentum=0.1, affine=True, track_running_stats=True):
#         super(ConvNet, self).__init__()
#         self.in_planes = opt['in_planes']
#         self.out_planes = opt['out_planes']
#         self.num_stages = opt['num_stages']
#         if type(self.out_planes) == int:
#             self.out_planes = [self.out_planes for i in range(self.num_stages)]
#         assert (type(self.out_planes) == list and len(self.out_planes) == self.num_stages)
#
#         num_planes = [self.in_planes, ] + self.out_planes
#         userelu = opt['userelu'] if ('userelu' in opt) else True
#
#         conv_blocks = []
#         for i in range(self.num_stages):
#             if i == (self.num_stages - 1):
#                 conv_blocks.append(
#                     ConvBlock(num_planes[i], num_planes[i + 1], userelu=userelu))
#             else:
#                 conv_blocks.append(
#                     ConvBlock(num_planes[i], num_planes[i + 1]))
#         self.conv_blocks = nn.Sequential(*conv_blocks)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         out = self.conv_blocks(x)
#         out = out.view(out.size(0), -1)
#         return out
#


class EmbeddingImagenet(nn.Module):
    def __init__(self):
        super(EmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        # self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden * 1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden * 1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden * 1.5),
                                              out_channels=self.hidden * 2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden * 2,
                                              out_channels=self.hidden * 4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.conv_5 = nn.Sequential(nn.Conv2d(in_channels=self.hidden * 4,
                                              out_channels=85,
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=85),
                                    nn.AdaptiveMaxPool2d((14, 14))
                                    )

    def forward(self, input_data):
        output_data = self.conv_5(self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_data)))))
        return output_data
