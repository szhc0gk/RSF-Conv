# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 22:26:10 2025

@author: szhc0gk
"""

from .RSFConvUnetParts import *
from math import ceil


class RSFConvUnet(nn.Module):
    def __init__(self, n_channels, n_classes, rotNum=8, initS=3, gapS=1.25, numS=4):
        super(RSFConvUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        scaleList = [initS * gapS ** i for i in range(numS)]
        rotScaleNum = numS * rotNum
        factor_channels = ceil(64 / rotScaleNum)
        self.iniG = InConv(n_channels, rotNum=rotNum, scaleList=scaleList)
        self.inc = DoubleConv(n_channels, factor_channels, rotNum=rotNum, scaleList=scaleList)
        self.down1 = Down(factor_channels, factor_channels * 2, rotNum=rotNum, scaleList=scaleList)
        self.down2 = Down(factor_channels * 2, factor_channels * 4, rotNum=rotNum, scaleList=scaleList)
        self.down3 = Down(factor_channels * 4, factor_channels * 8, rotNum=rotNum, scaleList=scaleList)
        self.down4 = Down(factor_channels * 8, factor_channels * 8, rotNum=rotNum, scaleList=scaleList)
        self.up1 = Up(factor_channels * 16, factor_channels * 4, rotNum=rotNum, scaleList=scaleList)
        self.up2 = Up(factor_channels * 8, factor_channels * 2, rotNum=rotNum, scaleList=scaleList)
        self.up3 = Up(factor_channels * 4, factor_channels, rotNum=rotNum, scaleList=scaleList)
        self.up4 = Up(factor_channels * 2, factor_channels, rotNum=rotNum, scaleList=scaleList)
        self.outc = OutConv(factor_channels, n_classes, rotNum=rotNum, scaleList=scaleList)

    def forward(self, x):
        x0 = self.iniG(x)
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return F.sigmoid(logits)
