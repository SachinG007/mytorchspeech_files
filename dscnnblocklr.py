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


class DSCNNBlockLR(torch.nn.Module):
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
            activation='sigmoid'):

        super(DSCNNBlockLR, self).__init__()

        activators = {
            'sigmoid': torch.nn.Sigmoid(),
            'relu': torch.nn.ReLU(),
            'leakyrelu': torch.nn.LeakyReLU(),
            'tanhgate': _TanhGate(),
            'none': None
        }

        if activation not in activators:
            raise ValueError('Available activations are: %s' % ', '.join(activators.keys()))

        if activation == 'tanhgate':
            in_channels = int(in_channels/2)

        nonlin = activators[activation]

        if batch_norm > 0.0:
            batch_block = torch.nn.BatchNorm1d(in_channels, affine=False, momentum=batch_norm)
        else:
            batch_block = None

        depth_cnn = torch.nn.Conv1d(in_channels, in_channels, kernel_size=kernel, stride=1, groups=in_channels)
        # point_cnn = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)
        self.G = torch.nn.Linear(in_channels, 50, bias=False)
        self.H = torch.nn.Linear(50, out_channels, bias=False)

        if shuffle != 0 and groups > 1:
            shuffler = _IndexSelect(in_channels, shuffle, groups)
        else:
            shuffler = None

        if avg_pool > 0:
            pool = torch.nn.AvgPool1d(kernel_size=avg_pool, stride=1)
        else:
            pool = None

        if dropout > 0:
            dropout_block = torch.nn.Dropout(p=dropout)
        else:
            dropout_block = None

        # seq = [nonlin, batch_block, depth_cnn, shuffler, point_cnn, dropout_block, pool]
        seq1 = [nonlin, batch_block, depth_cnn]#, shuffler, point_cnn, dropout_block, pool]
        seq_f1 = [item for item in seq1 if item is not None]
        if len(seq_f1) == 1:
            self._op1 = seq_f1[0]
        else:
            self._op1 = torch.nn.Sequential(*seq_f1)

        seq2 = [shuffler, dropout_block, pool]
        seq_f2 = [item for item in seq2 if item is not None]
        if len(seq_f2) == 1:
            print("Only 1 op in seq2")
            self._op2 = seq_f2[0]
        else:
            self._op2 = torch.nn.Sequential(*seq_f2)

    def forward(self, x):
        x = self._op1(x)
        x = x.permute(0,2,1) #NCL to NLC
        x = self.H(self.G(x))
        x = x.permute(0,2,1) #NLC to NCL
        x = self._op2(x)
        return x



class LR_conv(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', rank=50):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        
        assert kernel_size == 5
        rank =  rank
        self.W1 = Parameter(torch.Tensor(self.out_channels, rank))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        self.W2 = Parameter(torch.Tensor(rank, self.in_channels * 5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        self.weight = None

    def forward(self, input):
        lr_weight = torch.matmul(self.W1, self.W2)
        lr_weight = torch.reshape(lr_weight, (self.out_channels, self.in_channels, 5))
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            lr_weight, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, lr_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class DSCNNBlockLR_k5(torch.nn.Module):
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
            activation='sigmoid', rank=50):

        super(DSCNNBlockLR_k5, self).__init__()

        activators = {
            'sigmoid': torch.nn.Sigmoid(),
            'relu': torch.nn.ReLU(),
            'leakyrelu': torch.nn.LeakyReLU(),
            'tanhgate': _TanhGate(),
            'none': None
        }

        if activation not in activators:
            raise ValueError('Available activations are: %s' % ', '.join(activators.keys()))

        if activation == 'tanhgate':
            in_channels = int(in_channels/2)

        nonlin = activators[activation]

        if batch_norm > 0.0:
            batch_block = torch.nn.BatchNorm1d(in_channels, affine=False, momentum=batch_norm)
        else:
            batch_block = None

        #do_depth = False
        # depth_cnn = torch.nn.Conv1d(in_channels, in_channels, kernel_size=kernel, stride=1, groups=in_channels)
        depth_cnn = None
        # point_cnn = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)
        point_cnn = LR_conv(in_channels, out_channels, kernel_size=kernel, stride=stride, groups=groups, rank=rank, padding=2)

        if shuffle != 0 and groups > 1:
            shuffler = _IndexSelect(in_channels, shuffle, groups)
        else:
            shuffler = None

        if avg_pool > 0:
            pool = torch.nn.AvgPool1d(kernel_size=avg_pool, stride=1)
        else:
            pool = None

        if dropout > 0:
            dropout_block = torch.nn.Dropout(p=dropout)
        else:
            dropout_block = None

        seq1 = [nonlin, batch_block, depth_cnn, shuffler, point_cnn, dropout_block, pool]
        # seq1 = [nonlin, batch_block, depth_cnn]#, shuffler, point_cnn, dropout_block, pool]
        seq_f1 = [item for item in seq1 if item is not None]
        if len(seq_f1) == 1:
            self._op1 = seq_f1[0]
        else:
            self._op1 = torch.nn.Sequential(*seq_f1)

        seq2 = [shuffler, dropout_block, pool]
        seq_f2 = [item for item in seq2 if item is not None]
        if len(seq_f2) == 1:
            print("Only 1 op in seq2")
            self._op2 = seq_f2[0]
        else:
            self._op2 = torch.nn.Sequential(*seq_f2)

    def forward(self, x):
        # x = self._op1(x)
        # print(self._op1)
        # np.save('trace3/feature3.npy', torch.transpose(x[0],0,1).detach().cpu().numpy())
        # import pdb;pdb.set_trace()
        xb = self._op1[0](x)
        # stored_inp=torch.from_numpy(np.load('trace6/CNN1_outbnorm1.npy'))
        # stored_inp = stored_inp.unsqueeze(dim=0)
        # stored_inp = torch.transpose(stored_inp,2,1)
        # self._op1[1]=self._op1[1].cpu()
        # new_out=self._op1[1](stored_inp)
        # old_out=np.load('trace6/CNN1_outcnn1.npy')
        # import pdb;pdb.set_trace()
        x_cnn = self._op1[1](xb)
        # import pdb;pdb.set_trace()
        # print(self._op1[1].W1)
        # np.save('trace6/CNN1_outcnn1.npy', torch.transpose(x_cnn[1],0,1).detach().cpu().numpy())
        # np.save('trace6/CNN1_outbnorm1.npy', torch.transpose(xb[1],0,1).detach().cpu().numpy())
        # np.save('trace6/feature.npy', torch.transpose(x[1],0,1).detach().cpu().numpy())
        return x_cnn


class LR_pointcnn(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', rank=50):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        
        # assert stride == 3
        assert kernel_size == 1
        rank =  rank
        self.W1 = Parameter(torch.Tensor(self.out_channels, rank))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        self.W2 = Parameter(torch.Tensor(rank, self.in_channels))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        self.weight = None

    def forward(self, input):
        lr_weight = torch.matmul(self.W1, self.W2)
        lr_weight = torch.reshape(lr_weight, (self.out_channels, self.in_channels, 1))
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            lr_weight, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, lr_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class DSCNNBlockLR_better(torch.nn.Module):
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
            activation='sigmoid', rank=50):

        super(DSCNNBlockLR_better, self).__init__()

        activators = {
            'sigmoid': torch.nn.Sigmoid(),
            'relu': torch.nn.ReLU(),
            'leakyrelu': torch.nn.LeakyReLU(),
            'tanhgate': _TanhGate(),
            'none': None
        }

        if activation not in activators:
            raise ValueError('Available activations are: %s' % ', '.join(activators.keys()))

        if activation == 'tanhgate':
            in_channels = int(in_channels/2)

        nonlin = activators[activation]

        if batch_norm > 0.0:
            batch_block = torch.nn.BatchNorm1d(in_channels, affine=False, momentum=batch_norm)
        else:
            batch_block = None

        depth_cnn = torch.nn.Conv1d(in_channels, in_channels, kernel_size=kernel, stride=1, groups=in_channels, padding=2)
        point_cnn = LR_pointcnn(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, rank=rank)

        if shuffle != 0 and groups > 1:
            shuffler = _IndexSelect(in_channels, shuffle, groups)
        else:
            shuffler = None

        if avg_pool > 0:
            pool = torch.nn.AvgPool1d(kernel_size=avg_pool, stride=1)
        else:
            pool = None

        if dropout > 0:
            dropout_block = torch.nn.Dropout(p=dropout)
        else:
            dropout_block = None

        seq = [nonlin, batch_block, depth_cnn, shuffler, point_cnn, dropout_block, pool]
        # seq1 = [nonlin, batch_block, depth_cnn]#, shuffler, point_cnn, dropout_block, pool]
        seq_f = [item for item in seq if item is not None]
        if len(seq_f) == 1:
            self._op = seq_f[0]
        else:
            self._op = torch.nn.Sequential(*seq_f)

    def forward(self, x):
        x = self._op(x)
        return x


class SubBlock_R(torch.nn.Module):
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
        point_cnn = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)

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
        self.point_cnn3 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)

        self.batch_block1 = torch.nn.BatchNorm1d(out_channels, affine=False, momentum=batch_norm)
        self.batch_block2 = torch.nn.BatchNorm1d(out_channels, affine=False, momentum=batch_norm)
        self.batch_block3 = torch.nn.BatchNorm1d(out_channels, affine=False, momentum=batch_norm)

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
        
        x = self.point_cnn1(self.depth_cnn(x))
        side_x = self.batch_block2(self.point_cnn2(orig_x1))
        # import pdb;pdb.set_trace()
        x = x + side_x
        x = self.dropout_block(self.nonlin(x))

        outer_side_x = self.batch_block3(self.point_cnn3(orig_x2))
        x = x + outer_side_x
        return x

