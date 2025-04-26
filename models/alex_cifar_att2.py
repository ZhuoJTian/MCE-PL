import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, conv_layer, linear_layer, num_classes=10, k=1.0):
        super(AlexNet, self).__init__()
        self.conv1 = conv_layer(3, 64, kernel_size=5, stride=1, padding=1, bias=False, k=k)
        self.norm1 = nn.LocalResponseNorm(4, 2e-5, 0.75)
        self.pool1 = nn.MaxPool2d((3,3), (2,2), padding=0)

        self.conv2 = conv_layer(64, 128, kernel_size=5, stride=1, padding=1, bias=False, k=k)
        self.norm2 = nn.LocalResponseNorm(4, 1e-4, 0.75)
        self.pool2 = nn.MaxPool2d((3,3), (2,2), padding=0)

        self.conv3 = conv_layer(128, 256, kernel_size=5, stride=1, padding=1, bias=False, k=k)
        self.norm3 = nn.LocalResponseNorm(4, 1e-4, 0.75)
        self.pool3 = nn.MaxPool2d((3,3), (2,2), padding=(1,1))
        
        self.fc1 = linear_layer(1024, 192)
        self.fc2 = linear_layer(192, num_classes)
        self.k = k

    def forward(self, x, ini, mask_local, mask_neig):
        out = self.pool1(self.norm1(F.elu(self.conv1(x, mask_local[0], mask_neig[0]))))
        out = self.pool2(self.norm2(F.elu(self.conv2(out, mask_local[1], mask_neig[1]))))
        out = self.pool3(self.norm3(F.elu(self.conv3(out, mask_local[2], mask_neig[2]))))
        out = torch.flatten(out, 1)
        out = F.elu(self.fc1(out, mask_local[3], mask_neig[3]))
        out = F.elu(self.fc2(out, mask_local[4], mask_neig[4]))           
        return out