import torch
import torch.nn as nn
from torch.autograd import Variable
import torchspeech.nn.cnn as cnn
import rnn as rnn
import math


class BiFastGRNN(nn.Module):
    def __init__(self, inputDims, inputDims_b, hiddenDims, gate_nonlinearity,
                 update_nonlinearity, rank):
        super(BiFastGRNN, self).__init__()
        self.cell_fwd = rnn.FastGRNN(inputDims,
                                         hiddenDims,
                                         gate_nonlinearity,
                                         update_nonlinearity,
                                         batch_first=True,
                                         wRank=rank,
                                         uRank=rank)
        self.cell_bwd = rnn.FastGRNN(inputDims_b,
                                         100,
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


def X_preRNN_process(X):

    #FWD bricking
    brickLength = 60
    hopLength = 20
    X_bricked_f = X.unfold(1, brickLength, hopLength)
    X_bricked_f = X_bricked_f.permute(0, 1, 3, 2)
    #X_bricked_f [batch, num_bricks, brickLen, inpDim]
    oldShape_f = X_bricked_f.shape
    X_bricked_f = torch.reshape(
        X_bricked_f, [oldShape_f[0] * oldShape_f[1], oldShape_f[2], -1])
    # X_bricked_f [batch*num_bricks, brickLen, inpDim]

    #BWD bricking
    brickLength = 5
    hopLength = 1
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

    X_new_f = X_f[:, 0, :, :]  #batch,brickLen,hiddenDim
    X_new_f_rest = X_f[:, 1:, -20:, :]  #batch, numBricks-1, (hopLen),hiddenDim
    shape = X_new_f_rest.shape
    X_new_f_rest = torch.reshape(X_new_f_rest,
                                 [shape[0], shape[1] * shape[2], shape[3]])
    X_new_f = torch.cat((X_new_f, X_new_f_rest),
                        dim=1)  #batch,seqLen,hiddenDim
    #X_new_f [batch, seqLen, hiddenDim]

    #Backward Bricks folding
    X_b = torch.reshape(X_b, [oldShape_b[0], oldShape_b[1], oldShape_b[2], -1])
    #X_b [batch, num_bricks, brickLen, hiddenDim]
    X_b = torch.flip(
        X_b,
        [1])  #Reverse the ordering of the bricks (bring last brick to start)

    X_new_b = X_b[:, 0, :, :]  #batch,brickLen,inpDim
    X_new_b_rest = X_b[:, 1:,
                       -1:, :].squeeze(2)  #batch,(seqlen-brickLen),hiddenDim
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
                 rank=None):
        super(DSCNN_RNN_Block, self).__init__()

        self.cnn_channels = cnn_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.gate_nonlinearity = gate_nonlinearity
        self.update_nonlinearity = update_nonlinearity
        self.num_labels = num_labels
        self.device = device
        self.islstm = islstm
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
            cnn.DSCNNBlock(80,
                           cnn_channels,
                           5,
                           1,
                           1,
                           0,
                           0,
                           do_depth=False,
                           batch_norm=1e-2,
                           activation='none'),
            cnn.DSCNNBlock(cnn_channels,
                           cnn_channels,
                           batch_norm=1e-2,
                           dropout=0,
                           kernel=15,
                           stride=3,
                           activation='tanhgate'),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm1d(cnn_channels, affine=False, momentum=1e-2))

        self.RNN0 = BiFastGRNN(cnn_channels, cnn_channels, rnn_hidden_size,
                               self.gate_nonlinearity,
                               self.update_nonlinearity, rank)
        for i in range(1, self.rnn_num_layers):
            setattr(
                self, 'bnorm_f' + str(i),
                torch.nn.BatchNorm1d(rnn_hidden_size,
                                     affine=False,
                                     momentum=1e-2))
            setattr(
                self, 'bnorm_b' + str(i),
                torch.nn.BatchNorm1d(100,
                                     affine=False,
                                     momentum=1e-2))
            setattr(
                self, 'RNN' + str(i),
                BiFastGRNN(rnn_hidden_size, 100, rnn_hidden_size,
                           self.gate_nonlinearity, self.update_nonlinearity,
                           rank))
        self.CNN2 = torch.nn.Sequential(
            cnn.DSCNNBlock(100 + rnn_hidden_size,
                           41,
                           batch_norm=1e-2,
                           dropout=0,
                           kernel=5,
                           activation='tanhgate'))

    def forward(self, features, seqlen):

        batch, max_seq_len, _ = features.shape
        features = features.permute((0, 2, 1))  # NLC to NCL

        X = self.CNN1(features)  # Down to 30ms inference / 250ms window
        X = X.permute((0, 2, 1))  #  NCL to NLC

        max_seq_len = X.shape[1]
        seqlen /= 3
        seqlen = torch.clamp(seqlen, max=max_seq_len)
        X = X.contiguous()
        # print(X.shape)
        assert X.shape[1] % 20 == 0

        X_f, oldShape_f, X_b, oldShape_b = X_preRNN_process(X)
        #X [batch * num_bricks, brickLen, inpDim]
        X_b_f = torch.flip(X_b, [1])
        X_f, X_b = self.RNN0(X_f, X_b_f)
        for i in range(1, self.rnn_num_layers):
            X_f = X_f.permute((0, 2, 1))
            X_f = getattr(self, 'bnorm_f' + str(i))(X_f)
            X_f = X_f.permute((0, 2, 1))

            X_b = X_b.permute((0, 2, 1))
            X_b = getattr(self, 'bnorm_b' + str(i))(X_b)
            X_b = X_b.permute((0, 2, 1))

            X_f, X_b = getattr(self, 'RNN' + str(i))(X_f, X_b)
        X = X_postRNN_process(X_f, oldShape_f, X_b, oldShape_b)

            # print(X.shape)
        # print(f'X  {X}')

        # (batch, max_seq_len, num_directions, hidden_size)
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        X = X.view(batch, max_seq_len,
                   100 + self.rnn_hidden_size)

        # re-permute to get [batch, channels, max_seq_len/3 ]
        X = X.permute((0, 2, 1))  #  NLC to NCL
        # print("Pre CNN2 Shape : ", X.shape)
        X = self.CNN2(X)
        # print("Post CNN2  Shape : ", X.shape)
        # re-permute to get [batch, max_seq_len/3, 41]
        X = X.permute((0, 2, 1))  #  NCL to NLC

        return X

    def init_hidden(self, batch, rnn_hidden_size, rnn_num_layers):

        # the weights are of the form (batch, num_layers * num_directions , hidden_size)
        hidden = torch.randn(rnn_num_layers * self.direction_param, batch,
                             rnn_hidden_size)

        hidden = hidden.to(self.device)

        hidden = Variable(hidden)

        return hidden


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

        self.__name__ = 'Binary_Classification_Block'

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
            torch.nn.Dropout(self.dropout),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(self.direction_param * self.rnn_hidden_size, 2))

    def forward(self, features, seqlen):
        # print(f'features shape {features.shape}')
        # print(f'seqlen shape {seqlen}')
        self.middle_outputs = []
        self.middle_modules = []

        batch, max_seq_len, _ = features.shape

        if self.islstm:
            hidden1 = self.init_hidden(batch, self.rnn_hidden_size,
                                       self.rnn_num_layers)
            hidden2 = self.init_hidden(batch, self.rnn_hidden_size,
                                       self.rnn_num_layers)
        else:
            hidden = self.init_hidden(batch, self.rnn_hidden_size,
                                      self.rnn_num_layers)

            # print(f'hidden shape {hidden.shape}')

        features = features.permute((0, 2, 1))  # NLC to NCL

        if self.batch_assertion:
            X = features
            try:
                assert (not torch.any(torch.isnan(X)))
            except AssertionError as e:
                print('Nan found')
                raise e
            for i, frm in enumerate(self.CNN1):
                X = frm(X)
                try:
                    assert (not torch.any(torch.isnan(X)))
                except AssertionError as e:
                    print('Nan found')
                    raise e
                self.middle_outputs.append(X.clone().detach())
                self.middle_modules.append(frm)
        else:
            X = self.CNN1(features)  # Down to 30ms inference / 250ms window
        # X -> [batch, channels, max_seq_len/3]

        # re-permute to get [batch, max_seq_len/3, channels]
        X = X.permute((0, 2, 1))  #  NCL to NLC

        # print(f'X shape {X.shape}')
        max_seq_len = X.shape[1]

        # modify seqlen
        ######### seqlen /= 3    do not un comment because it already happened inplace before
        max_seq_len = min(torch.max(seqlen).item(), max_seq_len)
        seqlen = torch.clamp(seqlen, max=max_seq_len)
        self.seqlen = seqlen

        # pad according to seqlen
        X = torch.nn.utils.rnn.pack_padded_sequence(X,
                                                    seqlen,
                                                    batch_first=True,
                                                    enforce_sorted=False)

        if self.islstm:
            X, (hh, _) = self.RNN(X, (hidden1, hidden2))
        else:
            X, hh = self.RNN(X, hidden)

        # print(f'X  {X}')

        # (batch, max_seq_len, num_directions, hidden_size)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        X = X.view(batch, max_seq_len,
                   self.direction_param * self.rnn_hidden_size)

        if self.batch_assertion:
            self.middle_outputs.append(
                X.clone().detach()[:, :, :int(X.shape[2])])
            self.middle_modules.append(self.RNN)
            try:
                assert (not torch.any(torch.isnan(X)))
            except AssertionError as e:
                print('Nan found')
                raise

        X = X[torch.arange(batch).long(),
              seqlen.long() - 1, :].view(
                  batch, 1, self.direction_param * self.rnn_hidden_size)

        if self.batch_assertion:
            self.middle_outputs.append(
                X.clone().detach()[:, :, :int(X.shape[2])])
            self.middle_modules.append(self.RNN)
            try:
                assert (not torch.any(torch.isnan(X)))
            except AssertionError as e:
                print('Nan found')
                raise

        X = self.FCN(X)

        if self.batch_assertion:
            self.middle_outputs.append(X.clone().detach())
            self.middle_modules.append(self.FCN)
            try:
                assert (not torch.any(torch.isnan(X)))
            except AssertionError as e:
                print('Nan found')
                raise e

        return X

    def init_hidden(self, batch, rnn_hidden_size, rnn_num_layers):

        # the weights are of the form (batch, num_layers * num_directions , hidden_size)
        if self.batch_assertion:
            hidden = torch.zeros(rnn_num_layers * self.direction_param, batch,
                                 rnn_hidden_size)
        else:
            hidden = torch.randn(rnn_num_layers * self.direction_param, batch,
                                 rnn_hidden_size)

        hidden = hidden.to(self.device)

        hidden = Variable(hidden)

        return hidden

    def get_middle_computations(self):
        if self.batch_assertion:
            return self.middle_outputs, self.middle_modules, self.seqlen
        else:
            raise AssertionError('turn on assertion mode')