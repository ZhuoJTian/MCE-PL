import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers_att import GetSubnetUnstructured


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.ini = False
        self.k=0.7
        self.mu=0.7
    
    def set_ini(self, ini):
        self.ini = ini
    
    def set_k_mu(self, k, mu):
        self.k = k
        self.mu = mu

    def forward(self, x, ns_local, ns_neig, ns_p):
        if ns_local==0:
            self.conv1.set_ini(self.ini)
            self.conv1.set_k_mu(self.k, self.mu)
            out = F.relu(self.bn1(self.conv1(x, 0,0,0)))
            self.conv2.set_ini(self.ini)
            self.conv2.set_k_mu(self.k, self.mu)
            out = self.bn2(self.conv2(out, 0,0,0))            
        else:
            self.conv1.set_ini(self.ini)
            self.conv1.set_k_mu(self.k, self.mu)
            out = F.relu(self.bn1(self.conv1(x, ns_local[0], ns_neig[0], ns_p[0])))
            self.conv2.set_ini(self.ini)
            self.conv2.set_k_mu(self.k, self.mu)
            out = self.bn2(self.conv2(out, ns_local[1], ns_neig[1], ns_p[1]))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_layer(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


class GetSubnetFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        zeros = torch.zeros_like(scores).to(scores.device)
        ones = torch.ones_like(scores).to(scores.device)
        k_val = percentile(scores, (1 - k) * 100)
        return torch.where(scores < k_val, zeros, ones)

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class ResNet(nn.Module):
    def __init__(self, conv_layer, linear_layer, block, num_blocks, num_classes=10, k=1.0, unstructured=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv_layer = conv_layer
        self.k = k
        self.unstructured_pruning = unstructured

        self.conv1 = conv_layer(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = linear_layer(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.conv_layer, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, ini, ns_local, ns_neig, ns_p):                                                                                                               
        list_mu = [1.0, 1.0, 1.0, 1.0, 0.0]
        '''
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None and len(attr.shape)==4:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None and len(attr.shape)==4:
                            numel = attr.numel()
                            vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                            pointer += numel
        '''
        if ns_local==0:
            self.conv1.set_ini(ini)
            out = F.relu(self.bn1(self.conv1(x, 0, 0, 0)))
            i = 1
            for bs_block in self.layer1:
                bs_block.set_ini(ini)
                out = bs_block(out, 0, 0, 0)
                i = i + 2
            for bs_block in self.layer2:
                bs_block.set_ini(ini)
                out = bs_block(out, 0, 0, 0)
                i = i + 2
            for bs_block in self.layer3:
                bs_block.set_ini(ini)
                out = bs_block(out, 0, 0, 0)
                i = i + 2
            for bs_block in self.layer4:
                bs_block.set_ini(ini)
                out = bs_block(out, 0, 0, 0)
                i = i + 2
            # assert (i == len(ns_local))
            out = nn.AdaptiveAvgPool2d((1, 1))(out)
            out = out.view(out.size(0), -1)
            self.linear.set_ini(ini)
            out = self.linear(out)
        else:
            self.conv1.set_ini(ini)
            self.conv1.set_k_mu(0.2, list_mu[0])
            out = F.relu(self.bn1(self.conv1(x, ns_local[0], ns_neig[0], ns_p[0])))
            i = 1
            for bs_block in self.layer1:
                bs_block.set_ini(ini)
                bs_block.set_k_mu(0.2, list_mu[1])
                out = bs_block(out, ns_local[i: i + 2], ns_neig[i: i + 2], ns_p[i: i + 2])
                i = i + 2
            for bs_block in self.layer2:
                bs_block.set_ini(ini)
                bs_block.set_k_mu(0.2, list_mu[2])
                out = bs_block(out, ns_local[i: i + 2], ns_neig[i: i + 2], ns_p[i: i + 2])
                i = i + 2
            for bs_block in self.layer3:
                bs_block.set_ini(ini)
                bs_block.set_k_mu(0.2, list_mu[3])
                out = bs_block(out, ns_local[i: i + 2], ns_neig[i: i + 2], ns_p[i: i + 2])
                i = i + 2
            for bs_block in self.layer4:
                bs_block.set_ini(ini)
                bs_block.set_k_mu(0.2, list_mu[4])
                out = bs_block(out, ns_local[i: i + 2], ns_neig[i: i + 2], ns_p[i: i + 2])
                i = i + 2
            assert (i == len(ns_local))
            out = nn.AdaptiveAvgPool2d((1, 1))(out)
            out = out.view(out.size(0), -1)
            self.linear.set_ini(ini)
            out = self.linear(out)
        
        return out


def resnet18(conv_layer, linear_layer, **kwargs):
    return ResNet(conv_layer, linear_layer, BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(conv_layer, linear_layer, **kwargs):
    return ResNet(conv_layer, linear_layer, BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(conv_layer, linear_layer, **kwargs):
    return ResNet(conv_layer, linear_layer, Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(conv_layer, linear_layer, **kwargs):
    return ResNet(conv_layer, linear_layer, Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(conv_layer, linear_layer, **kwargs):
    return ResNet(conv_layer, linear_layer, Bottleneck, [3, 8, 36, 3], **kwargs)
