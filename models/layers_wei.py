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
        _, idx = scores.abs().flatten().sort()
        j = int((1 - k) * scores.numel())

        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        # flat_out[idx[j:]] = 1

        # take sum of dim-0
        wei_bi = torch.ones(scores.shape).to(out.device)
        wei_bi[abs(out)<=0.0001] = 0
        out_filter = torch.sum(wei_bi, dim=(1,2,3))
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
        _, idx = scores.abs().flatten().sort()
        j = int((1 - k) * scores.numel())

        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        # flat_out[idx[j:]] = 1

        # take sum of dim-0
        wei_bi = torch.ones(scores.shape).to(out.device)
        wei_bi[abs(scores)<=0.0001] = 0
        out_filter = torch.sum(wei_bi, dim=1)
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
            k = 1.0,
    ):
        super(SubnetConvUnstructured, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True
        self.weight_trans = 0
        self.k = k

    def forward(self, x):
        self.weight_trans = GetSubnetFil.apply(self.weight, self.k)
        x = F.conv2d(
                x, self.weight_trans, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        # nn.init.xavier_normal_(self.weight)
        # nn.init.constant_(self.bias, 0)
        self.weight.requires_grad = True
        self.bias.requires_grad = True
        self.weight_trans = 0
        self.k = 0.9

    def forward(self, x):
        self.weight_trans = GetSubnetLinear.apply(self.weight, self.k)
        x = F.linear(x, self.weight_trans, self.bias)
        return x
