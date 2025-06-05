# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 22:26:10 2025

@author: szhc0gk
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from PIL import Image
from math import ceil


class RSFconv(nn.Module):
    def __init__(self, inNum, outNum, rotNum, scaleList, inP=None, ifIni=0, bias=True, stride=1, group=1):
        super(RSFconv, self).__init__()

        self.inNum = inNum
        self.outNum = outNum
        self.rotNum = rotNum
        self.scaleList = scaleList
        self.stride = stride
        self.group = group
        self.biasflag = bias
        self.scaleNum = len(self.scaleList)
        self.disScaleList = [basisDiscrete(i) for i in self.scaleList]
        self.paddingList = [int((i - 1) // 2) for i in self.disScaleList]

        if inP is None:
            inP = ceil(self.scaleList[-1])

        if ifIni:
            self.rotExpand = 1
            self.scaleExpand = 1
        else:
            self.rotExpand = self.rotNum
            self.scaleExpand = self.scaleNum

        weights = Getinichange(inP, self.inNum * self.scaleExpand, self.outNum, self.rotExpand)
        self.weights = nn.Parameter(weights, requires_grad=True)

        for i in range(self.scaleNum):
            BasisC, BasisS = GetBasiscontinuechange(scaleList[i], rotNum, inP)
            self.register_buffer("Basis{}".format(i),
                                 torch.cat([BasisC, BasisS], 3))
        if self.biasflag:
            self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=True)
        else:
            self.c = 0.

    def forward(self, input):
        outlist = []
        bias = 0.

        for s in range(self.scaleNum):
            basis = eval('self.Basis{}'.format(s))
            tempW = torch.einsum('ijok,mnak->monaij', basis, self.weights)

            for r in range(self.rotExpand):
                rotgconv = np.hstack((np.arange(self.rotExpand - r, self.rotExpand), np.arange(self.rotExpand - r)))
                tempW[:, r, :, :, ...] = tempW[:, r, :, rotgconv, ...]

            if self.scaleExpand != 1:
                scalegconv = np.hstack((np.arange(self.scaleExpand - s, self.scaleExpand),
                                        np.arange(self.scaleExpand - s)))
                tempW = tempW.reshape(
                    [self.outNum, self.rotNum, self.scaleExpand, self.inNum, self.rotExpand, self.disScaleList[s],
                     self.disScaleList[s]])
                tempW = tempW.permute([0, 1, 3, 2, 4, 5, 6])
                for i in range(self.inNum):
                    tempW[:, :, i, :, ...] = tempW[:, :, i, scalegconv, ...]

            _filter = tempW.reshape(
                [self.outNum * self.rotNum, self.inNum * self.rotExpand * self.scaleExpand, self.disScaleList[s],
                 self.disScaleList[s]])

            if self.biasflag:
                bias = self.c.repeat([1, 1, self.rotNum, 1]).reshape([1, self.outNum * self.rotNum, 1, 1])

            out = f.conv2d(input, _filter,
                           padding=self.paddingList[s],
                           stride=self.stride,
                           dilation=1,
                           groups=self.group) + bias

            outlist.append(out)

        return torch.cat(outlist, dim=1)


def basisDiscrete(sizeP):
    return int(sizeP // 2 * 2 + 1)


def Getinichange(sizeP, inNum, outNum, expand):
    inX, inY, Mask = MaskCcontinue(sizeP)
    dissizeP = basisDiscrete(sizeP)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    X0 = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(X0, 0), 0), 4), 0)
    y = Y0[:, 1]
    y = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(y, 0), 0), 3), 0)
    orlW = np.zeros([outNum, inNum, expand, dissizeP, dissizeP, 1, 1])
    for i in range(outNum):
        for j in range(inNum):
            for k in range(expand):
                temp = np.array(
                    Image.fromarray(((np.random.randn(3, 3)) * 2.4495 / np.sqrt((inNum) * sizeP * sizeP))).resize(
                        (dissizeP, dissizeP)))
                orlW[i, j, k, :, :, 0, 0] = temp

    v = np.pi / sizeP * (sizeP - 1)
    k = np.reshape((np.arange(sizeP)), [1, 1, 1, 1, 1, sizeP, 1])
    l = np.reshape((np.arange(ceil(sizeP / 2))), [1, 1, 1, 1, 1, ceil(sizeP / 2)])
    tempA = np.sum(np.cos(k * v * X0) * orlW, 4) / sizeP
    tempB = -np.sum(np.sin(k * v * X0) * orlW, 4) / sizeP
    A = np.sum(np.cos(l * v * y) * tempA + np.sin(l * v * y) * tempB, 3) / sizeP
    B = np.sum(np.cos(l * v * y) * tempB - np.sin(l * v * y) * tempA, 3) / sizeP
    A = np.reshape(A, [outNum, inNum, expand, sizeP * ceil(sizeP / 2)])
    B = np.reshape(B, [outNum, inNum, expand, sizeP * ceil(sizeP / 2)])
    iniW = np.concatenate((A, B), axis=3)

    return torch.FloatTensor(iniW)


def MaskCcontinue(sizeP):
    sizePdis = basisDiscrete(sizeP)
    p = (sizePdis - 1) / 2
    x = np.arange(-p, p + 1) / p / sizeP * sizePdis
    X, Y = np.meshgrid(x, x)
    C = X ** 2 + Y ** 2
    Mask = np.exp(-np.maximum(C - 1, 0) / 0.2)

    return X, Y, Mask


def GetBasiscontinuechange(sizeP, tranNum=8, inP=None):
    sizePdis = basisDiscrete(sizeP)
    if inP == None:
        inP = sizePdis

    inX, inY, Mask = MaskCcontinue(sizeP)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    Mask = np.expand_dims(Mask, 2)
    theta = np.arange(tranNum) / tranNum * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)
    X = np.cos(theta) * X0 - np.sin(theta) * Y0
    Y = np.cos(theta) * Y0 + np.sin(theta) * X0
    X = np.expand_dims(np.expand_dims(X, 3), 4)
    Y = np.expand_dims(np.expand_dims(Y, 3), 4)
    v = np.pi / inP * (inP - 1)
    p = inP / 2

    k = np.reshape(np.arange(inP), [1, 1, 1, inP, 1])
    l = np.reshape(np.arange(ceil(inP / 2)), [1, 1, 1, 1, ceil(inP / 2)])

    BasisC = np.cos((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
    BasisS = np.sin((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
    BasisC = np.reshape(BasisC, [sizePdis, sizePdis, tranNum, inP * ceil(inP / 2)]) * \
             np.expand_dims(Mask, 3) / (sizeP ** 2)
    BasisS = np.reshape(BasisS, [sizePdis, sizePdis, tranNum, inP * ceil(inP / 2)]) * \
             np.expand_dims(Mask, 3) / (sizeP ** 2)

    BasisC = np.reshape(BasisC, [sizePdis, sizePdis, tranNum, inP * ceil(inP / 2)])
    BasisS = np.reshape(BasisS, [sizePdis, sizePdis, tranNum, inP * ceil(inP / 2)])

    return torch.FloatTensor(BasisC), torch.FloatTensor(BasisS)


class RSF_BN(nn.Module):
    def __init__(self, channels, rotNum, scaleList):
        super(RSF_BN, self).__init__()
        self.BN = nn.BatchNorm2d(channels)
        self.rsNum = rotNum * len(scaleList)

    def forward(self, X):
        X = self.BN(X.reshape([X.size(0), int(X.size(1) / self.rsNum), self.rsNum * X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.rsNum * X.size(1), int(X.size(2) / self.rsNum), X.size(3)])


class GroupPooling(nn.Module):
    def __init__(self, rotNum, scaleList):
        super(GroupPooling, self).__init__()
        self.rsNum = rotNum * len(scaleList)

    def forward(self, input):
        output = input.reshape([input.size(0), -1, self.rsNum, input.size(2), input.size(3)])
        output = torch.max(output, 2).values
        return output
