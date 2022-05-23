# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :CeNet
# @File     :heatmap
# @Date     :2022/3/7 21:00
# @Author   :Sun
# @Email    :szqqishi@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2


def generate_heatmap(heatmap,x, y, sigma):
    heatmap[x][y] = 4
    heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
    # am = np.amax(heatmap)
    # heatmap /= am / 255
    return heatmap

# target = np.zeros((64, 48))
# plt.imshow(target, cmap='hot', interpolation='nearest')
# plt.show()
# target = generate_heatmap(target, (7,7))
# plt.imshow(target, cmap='hot', interpolation='nearest')
# plt.show()
# print(target)
