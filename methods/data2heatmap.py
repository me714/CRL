# -*- coding: utf-8 -*-
# @Time    : 2022/6/11 16:09
# @Author  : Sun
# @FileName: data2heatmap.py


import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
import seaborn as sns
import uuid


def concept_concentrate(data):
    """
    shape of data : (75, 4, 25, 25)

    """
    avg_pool = torch.nn.AvgPool2d((6, 6), 6, padding=0)
    avg_tol_1 = torch.nn.AvgPool1d(6, 6)
    avg_tol_2 = torch.nn.AvgPool1d(6, 6)
    # print(data.size())
    concept = data.view(5, 15, 4, 25, 25)
    concept_map = torch.zeros([5, 4, 5, 5])
    concept_map[:, :, 4, 4] = concept[:, 1, :, 24, 24]
    concept_total_1 = concept[:, 1, :, 23:24, :24].view(5, 4, 24)
    concept_total_1 = avg_tol_1(concept_total_1)

    concept_total_2 = concept[:, 1, :, :24, 23:24].view(5, 4, 24)
    concept_total_2 = avg_tol_2(concept_total_2)


    concept_singal = concept[:, 1, :, :24, :24]
    concept_singal = avg_pool(concept_singal)
    concept_map[:, :, :4, :4] = concept_singal
    concept_map[:, :, 4, :4] = concept_total_1
    concept_map[:, :, :4, 4] = concept_total_2

    return concept_map

def plot_confusion_matrix(image, cm, labels_name, title):
    """
    image: [100, 3, 84, 84]
    cm: [5, 4, 5, 5]
    """
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    fig, ax = plt.subplots(5, 6)
    ax = ax.flatten()
    image = image.view(5, 20, 3, 84, 84)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.title(title)  # 图像标题
    j = 0
    for i in range(30):
        # fig.add_subplot(5, 5, i+1)

        if i % 6 == 0:
            image_1 = image[j, 1, :, :, :]
            if i == 0:
                ax[i].set_title('Image', fontsize=10)
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)
            img = image_1.swapaxes(0, 1)
            img = img.swapaxes(1, 2)
            img = (img * std) + mean
            # plt.imshow(img/255, interpolation='nearest')
            im = ax[i].imshow(img)
            ax[i].set_xticks([])  # 将标签印在x轴坐标上
            ax[i].set_yticks([])  # 将标签印在y轴坐标上

        else:
            if i == 1:
                ax[i].set_title('Head 1', fontsize=10)
            if i == 2:
                ax[i].set_title('Head 2', fontsize=10)
            if i == 3:
                ax[i].set_title('Head 3', fontsize=10)
            if i == 4:
                ax[i].set_title('Head 4', fontsize=10)
            if i == 5:
                ax[i].set_title('Head avg', fontsize=10)
            if i % 5 == 0:
                cm_1 = np.mean(cm[j, :, :, :], axis=0)
            else:
                cm_1 = cm[j, i % 4, :, :]
            # plt.imshow(cm_1, interpolation='nearest')
            # sns.heatmap(cm)
            im = ax[i].imshow(cm_1)
            num_local = np.array(range(len(labels_name)))
            ax[i].set_xticks(num_local, labels_name, rotation=90, fontsize=5)  # 将标签印在x轴坐标上
            ax[i].set_yticks(num_local, labels_name, fontsize=5)  # 将标签印在y轴坐标上


        if (i+1) % 6 == 0:
            j += 1
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    fig.colorbar(im, ax=ax)
    # cbr = fig.colorbar()
    # cbr.ax.tick_params(labelsize=5)

    # plt.subplot(541)
    # cm1 = cm[0, :, :]
    # plt.imshow(cm1, interpolation='nearest')    # 在特定的窗口上显示图像
    # # plt.title(title)    # 图像标题
    # cbr = plt.colorbar()
    # cbr.ax.tick_params(labelsize=5)
    # num_local = np.array(range(len(labels_name)))
    # plt.xticks(num_local, labels_name, rotation=90, fontsize=5)    # 将标签印在x轴坐标上
    # plt.yticks(num_local, labels_name, fontsize=5)    # 将标签印在y轴坐标上
    # plt.subplot(542)
    # cm2 = cm[1, :, :]
    # plt.imshow(cm2, interpolation='nearest')  # 在特定的窗口上显示图像
    # # plt.title(title)  # 图像标题
    # cbr = plt.colorbar()
    # cbr.ax.tick_params(labelsize=5)
    # num_local = np.array(range(len(labels_name)))
    # plt.xticks(num_local, labels_name, rotation=90, fontsize=5)  # 将标签印在x轴坐标上
    # plt.yticks(num_local, labels_name, fontsize=5)  # 将标签印在y轴坐标上
    # # plt.ylabel('True label')
    # # plt.xlabel('Predicted label')
    # plt.subplot(543)
    # cm3 = cm[2, :, :]
    # plt.imshow(cm2, interpolation='nearest')  # 在特定的窗口上显示图像
    # # plt.title(title)  # 图像标题
    # cbr = plt.colorbar()
    # cbr.ax.tick_params(labelsize=5)
    # num_local = np.array(range(len(labels_name)))
    # plt.xticks(num_local, labels_name, rotation=90, fontsize=5)  # 将标签印在x轴坐标上
    # plt.yticks(num_local, labels_name, fontsize=5)  # 将标签印在y轴坐标上
    # plt.subplot(544)
    # cm4 = cm[3, :, :]
    # plt.imshow(cm2, interpolation='nearest')  # 在特定的窗口上显示图像
    # # plt.title(title)  # 图像标题
    # cbr = plt.colorbar()
    # cbr.ax.tick_params(labelsize=5)
    # num_local = np.array(range(len(labels_name)))
    # plt.xticks(num_local, labels_name, rotation=90, fontsize=5)  # 将标签印在x轴坐标上
    # plt.yticks(num_local, labels_name, fontsize=5)  # 将标签印在y轴坐标上


def data2heatmap(image, attn_map):
    concept_map = concept_concentrate(attn_map).numpy()
    labels_name = ["a", "b", "c", "d", "e"]
    plot_confusion_matrix(image, concept_map, labels_name, "HAR Confusion Matrix")
    uuid_str = uuid.uuid4().hex
    img_path = 'D:\Projects\comet_concept\output\img_%s.jpg' % uuid_str
    plt.savefig(img_path, dpi=1000)
    # plt.show()



# if __name__ == "__main__":
#     data = torch.randn([75, 4, 25, 25])
#     image = torch.randn([100, 3, 84, 84])
#     concept_map = concept_concentrate(data).numpy()
#     # heatmap = sns.heatmap(data=concept_map)
#     # labels_name = ["钳喙", "轴节", "钳柄", "环柄", "全局"]
#     labels_name = ["a", "b", "c", "d", "e"]
#     plot_confusion_matrix(image, concept_map, labels_name, "HAR Confusion Matrix")
#     plt.show()