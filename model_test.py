# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :comet
# @File     :model_test
# @Date     :2022/3/31 21:07
# @Author   :Sun
# @Email    :szqqishi@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import torch

import backbone
from methods.comet import COMET

model = COMET(backbone.Conv6NP, n_way = 5, n_support = 5)
checkpoint = torch.load(r"D:\Projects\comet\SunSet1\checkpoints\CUB\Conv6NP_comet_0_5way_5shot\best_model.tar")

if __name__ == '__main__':
    # print(model)
    print(checkpoint)