# This code is modified from https://github.com/jakesnell/prototypical-networks

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from methods.relation_learner import WindowAttention
from methods.heatmap import generate_heatmap
from methods.focal_loss import FocalLossV1
from methods.resNet import CNN, BasicBlock
from backbone import Conv5
from torchvision import models


class COMET(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(COMET, self).__init__(model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.Linear1 = nn.Linear(49, 49)
        self.Linear2 = nn.Linear(49, 2)
        self.smoth = nn.SmoothL1Loss()
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.featurep = Conv5()

        self.wt = nn.Parameter(data=torch.ones(16), requires_grad=True).cuda()
        self.att = WindowAttention(64, 4)
        self.focal_loss = FocalLossV1()
        # self.cnn = load_pretrain()


    def set_forward(self, x, joints=None, is_feature=False):
        z_support, z_query, z_pos, attn, joints_label = self.parse_feature(x, joints, is_feature)

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        # focal_loss = FocalLossV1
        z_pos = z_pos.float()
        joints_label = joints_label.float()
        loss = self.smoth(z_pos, joints_label)
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores, attn, loss

    def set_forward_loss(self, x, joints):
        y_query1 = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query1 = Variable(y_query1.cuda())

        scores, attn, loss_reg = self.set_forward(x, joints)
        loss_trans = self.loss_fn(attn, y_query1.long())
        loss_class = self.loss_fn(scores, y_query1.long())
        loss = loss_reg + loss_class + loss_trans
        return loss

    def parse_feature(self, x, joints, is_feature):
        global z_pos, attn, joints_label
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x)
            z_feature = self.featurep(x)
            z_one = z_feature.view(self.n_way, self.n_support + self.n_query, -1)
            z_pos = z_all.view(z_all.size()[0], z_all.size()[1], -1)
            z_pos = self.Linear2(z_pos)
            joints = joints.contiguous().view(self.n_way * (self.n_support + self.n_query), *joints.size()[2:])
            img_len = x.size()[-1]
            feat_len = z_feature.size()[-1]
            joints[:, :, :2] = joints[:, :, :2] / img_len * feat_len
            joints = joints.round().int()
            joints_label = Variable(joints[:, :, 0:2].contiguous().cuda())
            joints_label = joints_label.repeat(1, 3, 1)
            joints = np.repeat(joints, 3, axis=1)
            batch_num = joints.size(0)
            # joints_num = joints.size(1)
            joints_num = 3
            z_pos = abs(z_pos).round().int()
            z_avg = self.globalpool(z_feature).view(z_feature.size(0), z_feature.size(1))
            feat_list = []
            for i in range(batch_num):
                feat = []
                for j in range(joints_num):
                    x = z_pos[i, j, 0]
                    y = z_pos[i, j, 1]
                    if joints[i, j, 2] == 1 and joints[i, j, 0] >= 0 and joints[i, j, 1] >= 0 and z_pos[
                        i, j, 0] < feat_len and z_pos[i, j, 1] < feat_len:
                        feature = z_feature[i, :, x, y]
                        feat.append(feature)
                    else:
                        feature = z_feature[i, :, x, y]
                        feat.append(feature)
                feature = z_avg[i, :]
                feat.append(feature)
                feat = torch.stack(feat, dim=0)
                feat_list.append(feat)
            feat_all = torch.stack(feat_list, dim=0)
            attn = self.att(feat_all)
            z_all = z_one.view(self.n_way, self.n_support + self.n_query, -1)

        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        return z_support, z_query, z_pos, attn, joints_label

    def correct(self, x, joints):
        scores, attn, loss = self.set_forward(x, joints, is_feature=False)
        y_query = np.repeat(range(self.n_way), self.n_query)
        scores = (scores + attn)/2
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, tf_writer):
        print_freq = 10

        avg_loss = 0
        for i, (x, y, joints) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x, joints)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                tf_writer.add_scalar('loss/train', avg_loss / float(i + 1), epoch)

    def test_loop(self, test_loader, record=None, return_std=False):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, y, joints) in enumerate(test_loader):
            # print('1')
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x, joints)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean


def load_pretrain():
    resnet18 = models.resnet18(pretrained=True)
    model = CNN(BasicBlock, [2, 2, 2, 2])
    pretrained_dict = resnet18.state_dict()
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)

    return model

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
