# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 22:26:10 2025

@author: szhc0gk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import Model.RSFConv as rsfn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, rotNum, scaleList, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.single_conv1 = nn.Sequential(
            rsfn.RSFconv(in_channels, mid_channels, rotNum, scaleList, bias=False),
            rsfn.RSF_BN(mid_channels, rotNum, scaleList),
            nn.ReLU(inplace=True)
        )
        self.single_conv2 = nn.Sequential(
            rsfn.RSFconv(mid_channels, out_channels, rotNum, scaleList, bias=False),
            rsfn.RSF_BN(out_channels, rotNum, scaleList),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.single_conv1(x)
        x = self.single_conv2(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, rotNum, scaleList):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, rotNum, scaleList)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, rotNum, scaleList):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, rotNum, scaleList)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, rotNum, scaleList):
        super(OutConv, self).__init__()
        self.conv = rsfn.RSFconv(in_channels, out_channels, rotNum, scaleList)
        self.groupEndConv = rsfn.GroupPooling(rotNum=rotNum, scaleList=scaleList)

    def forward(self, x):
        x = self.conv(x)
        x = self.groupEndConv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_channels, rotNum, scaleList):
        super(InConv, self).__init__()
        self.groupIniConv = rsfn.RSFconv(in_channels, in_channels,
                                         rotNum, scaleList, ifIni=1)

    def forward(self, x):
        return self.groupIniConv(x)
