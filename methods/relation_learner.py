# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :comet
# @File     :relation_learner
# @Date     :2022/2/26 15:58
# @Author   :Sun
# @Email    :szqqishi@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import torch
import torch.nn as nn
# from visualizer import get_local


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, support_num=1, query_num=16, c_num=5, trd=1, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.1):

        super().__init__()
        self.lookup_table_bias = nn.Parameter(torch.zeros(5, 5), requires_grad=True)
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.c_num = c_num
        self.proj = nn.Linear(self.c_num*self.c_num*self.dim, 2*self.c_num*self.c_num*self.dim)
        self.proj1 = nn.Linear(2*self.c_num*self.c_num*self.dim, 5)
        self.support_num = support_num
        self.act = nn.GELU()
        self.trd = trd
        self.query_num = query_num
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rel_h = nn.Parameter(torch.randn(self.dim // 2, self.c_num, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(self.dim // 2, 1, self.c_num), requires_grad=True)

    # @get_local('attn')
    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # 5*（5+16）, 25, 64 > 5, 16, 25, 64
        x = x.view(5, self.support_num + self.query_num, self.c_num * self.c_num, self.dim)[:, self.support_num:, :,
            :].contiguous()
        for i in range(self.trd):
            x1 = x
            x = x.view(5*self.query_num, self.c_num, self.c_num, self.dim).permute(0, 3, 1, 2)
            k_out_h, k_out_w = x.split(self.dim // 2, dim=1)
            x = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
            x = x.permute(0, 2, 3, 1).view(5*self.query_num, self.c_num*self.c_num, self.dim)
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = x + x1

        x = x.view(5*self.query_num, self.c_num*self.c_num*self.dim)
        x = self.proj(x)
        x = self.act(x)
        x = self.proj1(x)
        x = self.proj_drop(x)
        # x = x.view(5, (self.query_num+self.support_num), self.c_num*self.c_num*self.dim)
        return x