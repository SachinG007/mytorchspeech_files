import torch
import torch.nn as nn
from torch.autograd import Variable
import torchspeech.nn.cnn as cnn
import rnn as rnn
import math
from dscnnblocklr import *
import numpy as np 

class MatchBoxNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 num_labels=41):

        super(MatchBoxNet, self).__init__()

        self.declare_network(in_channels,num_labels)

        self.__name__ = 'MatchBoxNet'

    def declare_network(self, in_channels, num_labels):

        self.C1 = SubBlock_R(in_channels, 128, 11, stride=2)
        self.B1 = MainBlock_B(128, 64, 13)
        self.B2 = MainBlock_B(64, 64, 15)
        self.B3 = MainBlock_B(64, 64, 17)
        self.C2 = SubBlock_R(64, 128, 29, dilation=2)
        self.C3 = SubBlock_R(128, 128, 1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.C4 = torch.nn.Conv1d(128, num_labels, kernel_size=1, stride=1)

    def forward(self, features, seqlen):
        # import pdb;pdb.set_trace()
        # print("Features Size: ", features.size())        
        X = self.C1(features)
        # print("After C1: ", X.size())
        X = self.B3(self.B2(self.B1(X)))
        # print("After B: ", X.size())
        X = self.C4(self.pool(self.C3(self.C2(X))))
        # print("After C2: ", X.size())
        return X


class LSTMClassifier(torch.nn.Module):
    def __init__(self,
                 in_size,
                 rnn_hidden_size,
                 rnn_num_layers,
                 device,
                 islstm=False,
                 isBi=True,
                 momentum=1e-2,
                 num_labels=2,
                 dropout=0,
                 batch_assertion=False):
        super(LSTMClassifier, self).__init__()

        self.in_size = in_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.num_labels = num_labels
        self.device = device
        self.islstm = islstm
        self.isBi = isBi
        self.momentum = momentum
        self.dropout = dropout
        self.batch_assertion = batch_assertion

        if self.isBi:
            self.direction_param = 2
        else:
            self.direction_param = 1

        self.declare_network(in_size, rnn_hidden_size, rnn_num_layers,
                             num_labels)

        self.__name__ = 'LSTMClassifier'

    def declare_network(self, in_size, rnn_hidden_size, rnn_num_layers,
                        num_labels):

        self.RNN = nn.LSTM(input_size=in_size,
                            hidden_size=rnn_hidden_size,
                            num_layers=rnn_num_layers,
                            batch_first=True,
                            bidirectional=self.isBi)

        self.FCN = torch.nn.Sequential(
            torch.nn.Linear(self.direction_param * self.rnn_hidden_size,
                            num_labels))

    def forward(self, features, seqlen):

        batch, _, _ = features.shape

        # X = self.CNN1(features)  # Down to 30ms inference / 250ms window

        X = features.permute((0, 2, 1))  #  NCL to NLC

        max_seq_len = X.shape[1]

        max_seq_len = min(torch.max(seqlen).item(), max_seq_len)
        seqlen = torch.clamp(seqlen, max=max_seq_len)
        # import pdb;pdb.set_trace()
        self.seqlen = seqlen

        # pad according to seqlen
        X = torch.nn.utils.rnn.pack_padded_sequence(X,
                                                    seqlen,
                                                    batch_first=True,
                                                    enforce_sorted=False)

        X, (hh, _) = self.RNN(X)

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        X = X.view(batch, max_seq_len,
                   self.direction_param * self.rnn_hidden_size)

        X = X[torch.arange(batch).long(),
              seqlen.long() - 1, :].view(
                  batch, 1, self.direction_param * self.rnn_hidden_size)

        X = self.FCN(X)
        X = X.permute((0, 2, 1))  #  NLC to NCL

        return X

