import torch
import torch.nn as nn
from torch.autograd import Variable
import torchspeech.nn.cnn as cnn
import rnn as rnn
import math
from dscnnblocklr import DSCNNBlockLR_better, DSCNNBlockLR_k5, SubBlock_R, MainBlock_B
import numpy as np 

class BiFastGRNN(nn.Module):
    def __init__(self, inputDims, hiddenDims, gate_nonlinearity,
                 update_nonlinearity, rank):
        super(BiFastGRNN, self).__init__()
        self.cell_fwd = rnn.FastGRNNCUDA(inputDims,
                                         hiddenDims,
                                         gate_nonlinearity,
                                         update_nonlinearity,
                                         batch_first=True,
                                         wRank=rank,
                                         uRank=rank)
        self.cell_bwd = rnn.FastGRNNCUDA(inputDims,
                                         hiddenDims,
                                         gate_nonlinearity,
                                         update_nonlinearity,
                                         batch_first=True,
                                         wRank=rank,
                                         uRank=rank)

    def forward(self, input_f, input_b):
        # Bidirectional FastGRNN
        output1 = self.cell_fwd(input_f)
        output2 = self.cell_bwd(input_b)
        #Returning the flipped output only for the bwd pass
        #Will align it in the post processing
        return output1, output2


def X_preRNN_process(X, fwd_context):

    #FWD bricking
    # import pdb;pdb.set_trace()
    brickLength = fwd_context
    hopLength = 3
    X_bricked_f1 = X.unfold(1, brickLength, hopLength)
    X_bricked_f2 = X_bricked_f1.permute(0, 1, 3, 2)
    #X_bricked_f [batch, num_bricks, brickLen, inpDim]
    oldShape_f = X_bricked_f2.shape
    X_bricked_f = torch.reshape(
        X_bricked_f2, [oldShape_f[0] * oldShape_f[1], oldShape_f[2], -1])
    # X_bricked_f [batch*num_bricks, brickLen, inpDim]

    #BWD bricking
    brickLength = 9
    hopLength = 3
    X_bricked_b = X.unfold(1, brickLength, hopLength)
    X_bricked_b = X_bricked_b.permute(0, 1, 3, 2)
    #X_bricked_f [batch, num_bricks, brickLen, inpDim]
    oldShape_b = X_bricked_b.shape
    X_bricked_b = torch.reshape(
        X_bricked_b, [oldShape_b[0] * oldShape_b[1], oldShape_b[2], -1])
    # X_bricked_f [batch*num_bricks, brickLen, inpDim]
    return X_bricked_f, oldShape_f, X_bricked_b, oldShape_b


def X_postRNN_process(X_f, oldShape_f, X_b, oldShape_b):

    #Forward bricks folding
    X_f = torch.reshape(X_f, [oldShape_f[0], oldShape_f[1], oldShape_f[2], -1])
    #X_f [batch, num_bricks, brickLen, hiddenDim]
    # print("X_f shape ", X_f.shape)
    X_new_f = X_f[:, 0, ::3, :]  #batch,brickLen,hiddenDim
    # print("first brick samples ", X_new_f.shape)
    X_new_f_rest = X_f[:, :, -1, :].squeeze(2)  #batch, numBricks-1,hiddenDim
    # print("first brick samples ", X_new_f_rest.shape)
    shape = X_new_f_rest.shape
    # X_new_f_rest = torch.reshape(X_new_f_rest,
    #                              [shape[0], shape[1] * shape[2], shape[3]])
    X_new_f = torch.cat((X_new_f, X_new_f_rest),
                        dim=1)  #batch,seqLen,hiddenDim
    #X_new_f [batch, seqLen, hiddenDim]

    #Backward Bricks folding
    X_b = torch.reshape(X_b, [oldShape_b[0], oldShape_b[1], oldShape_b[2], -1])
    #X_b [batch, num_bricks, brickLen, hiddenDim]
    X_b = torch.flip(
        X_b,
        [1])  #Reverse the ordering of the bricks (bring last brick to start)

    X_new_b = X_b[:, 0, ::3, :]  #batch,brickLen,inpDim
    X_new_b_rest = X_b[:, :,
                       -1, :].squeeze(2)  #batch,(seqlen-brickLen),hiddenDim
    X_new_b = torch.cat((X_new_b, X_new_b_rest),
                        dim=1)  #batch,seqLen,hiddenDim
    X_new_b = torch.flip(X_new_b, [1])  #inverting the flip operation

    X_new = torch.cat((X_new_f, X_new_b), dim=2)  #batch,seqLen,2*hiddenDim
    return X_new


class DSCNN_RNN_Block(torch.nn.Module):
    def __init__(self,
                 cnn_channels,
                 rnn_hidden_size,
                 rnn_num_layers,
                 device,
                 gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh",
                 islstm=False,
                 isBi=True,
                 num_labels=41,
                 rank=None,
                 fwd_context=15):
        super(DSCNN_RNN_Block, self).__init__()

        self.cnn_channels = cnn_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.gate_nonlinearity = gate_nonlinearity
        self.update_nonlinearity = update_nonlinearity
        self.num_labels = num_labels
        self.device = device
        self.islstm = islstm
        self.fwd_context = fwd_context
        self.isBi = isBi
        if self.isBi:
            self.direction_param = 2
        else:
            self.direction_param = 1

        self.declare_network(cnn_channels, rnn_hidden_size, rnn_num_layers,
                             num_labels, rank)

        self.__name__ = 'DSCNN_RNN_Block'

    def declare_network(self, cnn_channels, rnn_hidden_size, rnn_num_layers,
                        num_labels, rank):

        self.CNN1 = torch.nn.Sequential(
            DSCNNBlockLR_k5(80,
                           cnn_channels,
                           5,
                           1,   
                           1,
                           0,
                           0,
                           do_depth=False,
                           batch_norm=1e-2,
                           activation='none', rank=rank))
        #torch tanh layer directly in the forward pass 
        self.bnorm_rnn = torch.nn.BatchNorm1d(cnn_channels, affine=False, momentum=1e-2)
        self.RNN0 = BiFastGRNN(cnn_channels, rnn_hidden_size,
                               self.gate_nonlinearity,
                               self.update_nonlinearity, rank)
        
        self.CNN2 = DSCNNBlockLR_better(2 * rnn_hidden_size,
                    2 * rnn_hidden_size,
                    batch_norm=1e-2,
                    dropout=0,
                    kernel=5,
                    activation='tanhgate', rank=rank)
        self.CNN3 = DSCNNBlockLR_better(2 * rnn_hidden_size,
                    2 * rnn_hidden_size,
                    batch_norm=1e-2,
                    dropout=0,
                    kernel=5,
                    activation='tanhgate', rank=rank)
        self.CNN4 = DSCNNBlockLR_better(2 * rnn_hidden_size,
                    2 * rnn_hidden_size,
                    batch_norm=1e-2,
                    dropout=0,
                    kernel=5,
                    activation='tanhgate', rank=rank)
        
        self.CNN5 = DSCNNBlockLR_better(2 * rnn_hidden_size,
                           41,
                           batch_norm=1e-2,
                           dropout=0,
                           kernel=5,
                           activation='tanhgate', rank=rank)

    def forward(self, features):
        
        # np.save('trace3/inp_feature3.npy', torch.transpose(features[0],0,1).detach().cpu().numpy())
        # import pdb;pdb.set_trace()
        batch, _, max_seq_len = features.shape
        # import pdb;pdb.set_trace()
        X = self.CNN1(features)  # Down to 30ms inference / 250ms window
        # np.save('trace3/X_outCNN1.npy', torch.transpose(X[0],0,1).detach().cpu().numpy())
        X = torch.tanh(X)
        X = self.bnorm_rnn(X)
        X = X.permute((0, 2, 1))  #  NCL to NLC

        # max_seq_len = X.shape[1]
        # seqlen /= 3
        # seqlen = torch.clamp(seqlen, max=max_seq_len)
        # import pdb;pdb.set_trace()
        X = X.contiguous()
        # print(X.shape)
        assert X.shape[1] % 3 == 0
        # import pdb;pdb.set_trace()
        X_f, oldShape_f, X_b, oldShape_b = X_preRNN_process(X, self.fwd_context)
        #X [batch * num_bricks, brickLen, inpDim]
        X_b_f = torch.flip(X_b, [1])
        
        # np.save('trace_rnn/X_inpRNN.npy', X_f[1].detach().cpu().numpy())
        
        X_f, X_b = self.RNN0(X_f, X_b_f)
        
        # np.save('trace_rnn_cnn/X_outRNN.npy', X_f[1].detach().cpu().numpy())
        X = X_postRNN_process(X_f, oldShape_f, X_b, oldShape_b)
        # print(X.shape)
        # print(f'X  {X}')

        # (batch, max_seq_len, num_directions, hidden_size)
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # X = X.view(batch, max_seq_len, 2* self.rnn_hidden_size)

        # re-permute to get [batch, channels, max_seq_len/3 ]
        X = X.permute((0, 2, 1))  #  NLC to NCL
        # print("Pre CNN2 Shape : ", X.shape)
        # np.save('trace7/X_inpCNN2.npy', torch.transpose(X[1],0,1).detach().cpu().numpy())
        X = self.CNN2(X)
        # np.save('trace7/X_outCNN2.npy', torch.transpose(X[1],0,1).detach().cpu().numpy())
        X = self.CNN3(X)
        # np.save('trace7/X_outCNN3.npy', torch.transpose(X[1],0,1).detach().cpu().numpy())
        X = self.CNN4(X)
        # np.save('trace7/X_outCNN4.npy', torch.transpose(X[1],0,1).detach().cpu().numpy())
        X = self.CNN5(X)
        # np.save('trace_rnn_cnn/X_outfinal.npy', torch.transpose(X[-2],0,1).detach().cpu().numpy())
        # a = np.load('trace_rnn_cnn/X_outfinal.npy')
        # print(a)
        # print("Post CNN2  Shape : ", X.shape)
        # re-permute to get [batch, max_seq_len/3, 41]
        # X = X.permute((0, 2, 1))  #  NCL to NLC

        return X


class Binary_Classification_Block(torch.nn.Module):
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
        super(Binary_Classification_Block, self).__init__()

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

        self.__name__ = 'Binary_Classification_Block_2lay'

    def declare_network(self, in_size, rnn_hidden_size, rnn_num_layers,
                        num_labels):

        self.CNN1 = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm1d(in_size, affine=False,
                                 momentum=self.momentum),
            torch.nn.Dropout(self.dropout))

        if self.islstm:
            self.RNN = nn.LSTM(input_size=in_size,
                               hidden_size=rnn_hidden_size,
                               num_layers=rnn_num_layers,
                               batch_first=True,
                               bidirectional=self.isBi)
        else:
            self.RNN = nn.GRU(input_size=in_size,
                              hidden_size=rnn_hidden_size,
                              num_layers=rnn_num_layers,
                              batch_first=True,
                              bidirectional=self.isBi)

        self.FCN = torch.nn.Sequential(
            torch.nn.Linear(self.direction_param * self.rnn_hidden_size,
                            num_labels))

    def forward(self, features, seqlen):

        batch, _, _ = features.shape

        hidden1 = self.init_hidden(batch, self.rnn_hidden_size,
                                    self.rnn_num_layers)
        hidden2 = self.init_hidden(batch, self.rnn_hidden_size,
                                    self.rnn_num_layers)

        X = self.CNN1(features)  # Down to 30ms inference / 250ms window

        X = X.permute((0, 2, 1))  #  NCL to NLC

        max_seq_len = X.shape[1]

        # modify seqlen
        ######### seqlen /= 3    do not un comment because it already happened inplace before
        max_seq_len = min(torch.max(seqlen).item(), max_seq_len)
        seqlen = torch.clamp(seqlen, max=max_seq_len)
        # import pdb;pdb.set_trace()
        self.seqlen = seqlen

        # pad according to seqlen
        X = torch.nn.utils.rnn.pack_padded_sequence(X,
                                                    seqlen,
                                                    batch_first=True,
                                                    enforce_sorted=False)

        X, (hh, _) = self.RNN(X, (hidden1, hidden2))

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        X = X.view(batch, max_seq_len,
                   self.direction_param * self.rnn_hidden_size)

        X = X[torch.arange(batch).long(),
              seqlen.long() - 1, :].view(
                  batch, 1, self.direction_param * self.rnn_hidden_size)

        X = self.FCN(X)

        return X

    def init_hidden(self, batch, rnn_hidden_size, rnn_num_layers):

        # the weights are of the form (batch, num_layers * num_directions , hidden_size)
        if self.batch_assertion:
            hidden = torch.zeros(rnn_num_layers * self.direction_param, batch,
                                 rnn_hidden_size)
        else:
            hidden = torch.zeros(rnn_num_layers * self.direction_param, batch,
                                 rnn_hidden_size)

        hidden = hidden.to(self.device)

        hidden = Variable(hidden)

        return hidden

    def get_middle_computations(self):
        if self.batch_assertion:
            return self.middle_outputs, self.middle_modules, self.seqlen
        else:
            raise AssertionError('turn on assertion mode')

class Binary_Classification_Block_withconv(torch.nn.Module):
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
        super(Binary_Classification_Block_withconv, self).__init__()

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

        self.__name__ = 'Binary_Classification_Block_conv'

    def declare_network(self, in_size, rnn_hidden_size, rnn_num_layers,
                        num_labels):

        self.CNN1 = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm1d(41, affine=False,
                                 momentum=self.momentum),
            torch.nn.Dropout(self.dropout),
            SubBlock_R(41,128,11),
            MainBlock_B(128, 64, 13, repeat=4),
            MainBlock_B(64, 64, 13, repeat=4),
            MainBlock_B(64, in_size, 13, repeat=4))

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

        hidden1 = self.init_hidden(batch, self.rnn_hidden_size,
                                    self.rnn_num_layers)
        hidden2 = self.init_hidden(batch, self.rnn_hidden_size,
                                    self.rnn_num_layers)

        X = self.CNN1(features)  # Down to 30ms inference / 250ms window

        X = X.permute((0, 2, 1))  #  NCL to NLC

        max_seq_len = X.shape[1]

        # modify seqlen
        ######### seqlen /= 3    do not un comment because it already happened inplace before
        max_seq_len = min(torch.max(seqlen).item(), max_seq_len)
        seqlen = torch.clamp(seqlen, max=max_seq_len)
        # import pdb;pdb.set_trace()
        self.seqlen = seqlen

        # pad according to seqlen
        X = torch.nn.utils.rnn.pack_padded_sequence(X,
                                                    seqlen,
                                                    batch_first=True,
                                                    enforce_sorted=False)

        X, (hh, _) = self.RNN(X, (hidden1, hidden2))

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        X = X.view(batch, max_seq_len,
                   self.direction_param * self.rnn_hidden_size)

        X = X[torch.arange(batch).long(),
              seqlen.long() - 1, :].view(
                  batch, 1, self.direction_param * self.rnn_hidden_size)

        X = self.FCN(X)

        return X

    def init_hidden(self, batch, rnn_hidden_size, rnn_num_layers):

        # the weights are of the form (batch, num_layers * num_directions , hidden_size)
        if self.batch_assertion:
            hidden = torch.zeros(rnn_num_layers * self.direction_param, batch,
                                 rnn_hidden_size)
        else:
            hidden = torch.zeros(rnn_num_layers * self.direction_param, batch,
                                 rnn_hidden_size)

        hidden = hidden.to(self.device)

        hidden = Variable(hidden)

        return hidden

class CRNN(torch.nn.Module):
    def __init__(self,
                 in_steps,
                 in_mels,
                 conv_channels,
                 conv_kernel_size,
                 conv_stride,
                 rnn_hidden_size,
                 rnn_num_layers,
                 fc_hidden_dim,
                 device,
                 isBi=True,
                 momentum=1e-2,
                 num_labels=2,
                 dropout=0,
                 batch_assertion=False):
        super(CRNN, self).__init__()

        self.in_steps = in_steps
        self.in_mels = in_mels
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size    # (Freq, Time)
        self.conv_stride = conv_stride              # (Freq, Time)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.num_labels = num_labels
        self.device = device
        self.isBi = isBi
        self.momentum = momentum
        self.dropout = dropout
        self.batch_assertion = batch_assertion

        if self.isBi:
            self.direction_param = 2
        else:
            self.direction_param = 1
        # Input = (*, C, H, W) = (*, C, Freq, Time)
        rnn_in = int(conv_channels * int((in_mels - conv_kernel_size[0])/conv_stride[0] + 1) )
        rnn_steps = int((in_steps - conv_kernel_size[1])/conv_stride[1] + 1)
        print(rnn_steps, 2*rnn_hidden_size)
        self.rnn_in = rnn_in
        self.rnn_steps = rnn_steps
        self.conv = nn.Conv2d(1, conv_channels, kernel_size = conv_kernel_size, stride = conv_stride)
        self.gru = nn.GRU(input_size=rnn_in,
                              hidden_size=rnn_hidden_size,
                              num_layers=rnn_num_layers,
                              batch_first=True,
                              bidirectional=self.isBi)
        self.fc = nn.Linear(self.direction_param * rnn_hidden_size * rnn_steps, fc_hidden_dim)
        self.out_layer = nn.Linear(fc_hidden_dim, num_labels)
        self.__name__ = 'CRNN'

    def forward(self, X):
        X = torch.unsqueeze(X, 1)
        print(X.shape)
        X = self.conv(X)
        X = nn.ReLU()(X)
        print(X.shape)
        X = X.permute(0,3,1,2)
        X_shape = X.shape
        print(X.shape)
        X = torch.reshape(X, (X_shape[0], X_shape[1], X_shape[2] * X_shape[3]))
        print(X.shape)
        X, temp = self.gru(X)
        X = nn.ReLU()(X)
        X = nn.Flatten()(X)
        print(X.shape)
        X = self.fc(X)
        X = nn.ReLU()(X)
        print(X.shape)
        X = self.out_layer(X)
        return X