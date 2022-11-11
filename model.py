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
