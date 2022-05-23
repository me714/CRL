# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.proto = ProtoModule([64, 7, 7])


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, *self.feat_dim)#the shape of z is [n_data, n_dim]
        z_proto = torch.transpose(z_proto, 1, 2)
        z_proto = self.proto(z_proto)
        z_proto = z_proto.view(self.n_way, -1)
        z_query     = z_query.contiguous().view(self.n_way * self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query.long())


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)



class RelationConvBlock(nn.Module):
    def __init__(self, indim, outdim, kersize, padding = 0):
        super(RelationConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C      = nn.Conv3d(indim, outdim, kersize, padding = padding )
        self.BN     = nn.BatchNorm3d(outdim, momentum=1, affine=True)
        self.relu   = nn.ReLU()
        # self.pool   = nn.MaxPool3d(2)

        self.parametrized_layers = [self.C, self.BN, self.relu]

        for layer in self.parametrized_layers:
            backbone.init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self,x):
        out = self.trunk(x)
        return out

class ProtoModule(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size):
        super(ProtoModule, self).__init__()

        # self.loss_type = loss_type
        padding = 1 if ( input_size[1] <10 ) and ( input_size[2] < 10 ) else 0 # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling

        self.layer1 = RelationConvBlock(input_size[0], input_size[0]*2, [5, 1, 1])
        self.layer2 = RelationConvBlock(input_size[0]*2, input_size[0], [1, 1, 1])

        # shrink_s = lambda s: int((int((s - 2 + 2*padding)/2)-2 + 2*padding)/2)

        # self.fc1 = nn.Linear( input_size[0]* shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size )
        # self.fc2 = nn.Linear( hidden_size, 1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = out.view(out.size(0),-1)
        # out = F.relu(self.fc1(out))
        # if self.loss_type == 'mse':
        #     out = F.sigmoid(self.fc2(out))
        # elif self.loss_type == 'softmax':
        #     out = self.fc2(out)

        return out
