import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import copy

class GetSubnetFil(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        # take sum of dim-0
        out_filter = torch.sum(out, dim=(1,2,3))
        out_filter[out_filter <= 20] = 0
        out_filter[out_filter > 20] = 1
        out_filter = out_filter.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,out.shape[1], out.shape[2], out.shape[3])
        out = out_filter * out

        return out

    @staticmethod
    def backward(ctx, g):
        return g, None


class GetSubnetLinear(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        # take sum of dim-0
        out_filter = torch.sum(out, dim=1)
        out_filter[out_filter <= 20] = 0
        out_filter[out_filter > 20] = 1
        out_filter = out_filter.unsqueeze(-1).repeat(1, out.shape[1])
        out = out_filter * out

        return out

    @staticmethod
    def backward(ctx, g):
        return g, None
    

class SubnetConvUnstructured(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            k=1.0,
    ):
        super(SubnetConvUnstructured, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias
        )
        self.popup_scores_local = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores_local, a=math.sqrt(5))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0
        self.k = k
        self.mask = 0

    def get_w(self):
        attr1 = self.popup_scores
        adj = GetSubnetFil.apply(attr1.abs(), self.k)

        attr2 = self.weight
        numel = attr2.numel()
        self.w = attr2 * adj[0: numel].view_as(attr2)
        return adj[0: numel].view_as(attr2), self.w

    def forward(self, x):
        self.popup_scores = 1.0*self.popup_scores_local
        self.mask, self.w = self.get_w()
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        self.popup_scores_local = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores_local, a=math.sqrt(5))
        self.k = 0.9
        self.mask = 0

    def set_prune_rate(self, k):
        self.k = k

    def set_ini(self, ini):
        self.ini = ini

    def get_w(self):
        attr1 = self.popup_scores
        adj = GetSubnetLinear.apply(attr1.abs(), self.k)

        attr2 = self.weight
        numel = attr2.numel()
        self.w = attr2 * adj[0: numel].view_as(attr2)
        return adj.view_as(attr2), self.w

    def forward(self, x):
        self.popup_scores = 1.0*self.popup_scores_local
        self.mask, self.w = self.get_w()
        x = F.linear(x, self.w, self.bias)
        return x
