"""Depth CNN module"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import _single
import math
import numpy as np 

class _IndexSelect(torch.nn.Module):
    """Channel permutation module. The purpose of this is to allow mixing across the CNN groups."""
    def __init__(self, channels, direction, groups):
        super(_IndexSelect, self).__init__()

        if channels % groups != 0:
            raise ValueError('Channels should be a multiple of the groups')

        self._index = torch.zeros((channels), dtype=torch.int64)
        count = 0

        if direction > 0:
            for gidx in range(groups):
                for nidx in range(gidx, channels, groups):
                    self._index[count] = nidx
                    count += 1
        else:
            for gidx in range(groups):
                for nidx in range(gidx, channels, groups):
                    self._index[nidx] = count
                    count += 1

    def forward(self, value):
        if value.device != self._index.device:
            self._index = self._index.to(value.device)

        return torch.index_select(value, 1, self._index)


class _TanhGate(torch.nn.Module):
    def __init__(self):
        super(_TanhGate, self).__init__()

    def forward(self, value):
        channels = value.shape[1]
        piv = int(channels/2)

        sig_data = value[:, 0:piv, :]
        tanh_data = value[:, piv:, :]

        sig_data = torch.sigmoid(sig_data)
        tanh_data = torch.tanh(tanh_data)
        return sig_data * tanh_data


class SubBlock_R(torch.nn.Module):
    """A depth-separate CNN block"""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            dilation=1,
            groups=1,
            avg_pool=2,
            dropout=0.1,
            batch_norm=0.1,
            do_depth=True,
            shuffle=0,
            activation='relu'):

        super(SubBlock_R, self).__init__()

        activators = {
            'sigmoid': torch.nn.Sigmoid(),
            'relu': torch.nn.ReLU(),
            'leakyrelu': torch.nn.LeakyReLU(),
            'tanhgate': _TanhGate(),
            'none': None
        }

        depth_cnn = torch.nn.Conv1d(in_channels, in_channels, kernel_size=kernel, stride=1, groups=in_channels, padding=int((kernel-1)/2))
        point_cnn = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, dilation=dilation)

        batch_block = torch.nn.BatchNorm1d(out_channels, affine=False, momentum=batch_norm)

        nonlin = activators[activation]

        dropout_block = torch.nn.Dropout(p=dropout)

        seq1 = [depth_cnn, point_cnn, batch_block, nonlin, dropout_block]
        
        self._op1 = torch.nn.Sequential(*seq1)

    def forward(self, x):
        x = self._op1(x)
        return x

class MainBlock_B(torch.nn.Module):
    """A depth-separate CNN block"""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            groups=1,
            avg_pool=2,
            dropout=0.1,
            batch_norm=0.1,
            do_depth=True,
            shuffle=0,
            activation='relu',
            repeat=2):

        super(MainBlock_B, self).__init__()

        activators = {
            'sigmoid': torch.nn.Sigmoid(),
            'relu': torch.nn.ReLU(),
            'leakyrelu': torch.nn.LeakyReLU(),
            'tanhgate': _TanhGate(),
            'none': None
        }

        self.depth_cnn = torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel, stride=1, groups=out_channels, padding=int((kernel-1)/2))
        self.point_cnn1 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=stride, groups=groups)
        self.point_cnn2 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)
        # self.point_cnn3 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)

        self.batch_block1 = torch.nn.BatchNorm1d(out_channels, affine=False, momentum=batch_norm)
        self.batch_block2 = torch.nn.BatchNorm1d(out_channels, affine=False, momentum=batch_norm)
        # self.batch_block3 = torch.nn.BatchNorm1d(out_channels, affine=False, momentum=batch_norm)

        self.nonlin = activators[activation]

        self.dropout_block = torch.nn.Dropout(p=dropout)
        
        self.repeat = repeat
        for i in range(self.repeat-1):
            if i==0:
                setattr(self, 'sub_block' + str(i), SubBlock_R(in_channels,out_channels,kernel))
            else:
                setattr(self, 'sub_block' + str(i), SubBlock_R(out_channels,out_channels,kernel))

    def forward(self, x):
        orig_x1 = x.clone()
        orig_x2 = x.clone()
        for i in range(self.repeat-1):
            x = getattr(self,'sub_block'+str(i))(x)
        
        x = self.batch_block1(self.point_cnn1(self.depth_cnn(x)))
        side_x = self.batch_block2(self.point_cnn2(orig_x1))
        # import pdb;pdb.set_trace()
        x = x + side_x
        x = self.dropout_block(self.nonlin(x))
        return x

