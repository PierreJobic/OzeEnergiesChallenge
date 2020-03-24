# coding: UTF-8
"""This module defines the different networks implemented.

Example
-------
$ net = BenchmarkLSTM()

"""

import torch
import torch.nn as nn

class BenchmarkLSTM(nn.Module):
    """Example network for solving Oze datachallenge.

    Attributes
    ----------
    lstm: Torch LSTM
        LSTM layers.
    linear: Torch Linear
        Fully connected layer.
    """

    def __init__(self, input_dim=38, hidden_dim=100, output_dim=8, num_layers=3, **kwargs):
        """Defines LSTM and Linear layers.

        Parameters
        ----------
        input_dim: int, optional
            Input dimension. Default is 37 (see challenge description).
        hidden_dim: int, optional
            Latent dimension. Default is 100.
        output_dim: int, optional
            Output dimension. Default is 8 (see challenge description).
        num_layers: int, optional
            Number of LSTM layers. Default is 3.
        """
        super().__init__(**kwargs)

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Propagate input through the network.

        Parameters
        m = Batch_Size
        K = length of time series = 672 (a month hourly)
        ----------
        x: Tensor
            Input tensor with shape (m, K, input_dim)

        Returns
        -------
        output: Tensor
            Output tensor with shape (m, K, output_dim)
        """
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out)
        return output

class TestRecurrentNet(nn.Module):
    def __init__(self, Type='gru', input_dim=38, hidden_dim=100, output_dim=8, 
                 num_layers=3, 
                 batch_first=True, 
                 bidirectional=False, 
                 **kwargs):
        super(TestRecurrentNet, self).__init__()
        print('type is among (lstm, rnn, gru):', Type)
        if Type == 'lstm':
            self.net = nn.LSTM(input_dim, hidden_dim, 
                               num_layers=num_layers, 
                               bidirectional=bidirectional, 
                               batch_first=batch_first, 
                               **kwargs)
        if Type == 'rnn':
            self.net = nn.RNN(input_dim, hidden_dim, 
                              num_layers=num_layers, 
                              bidirectional=bidirectional, 
                              batch_first=batch_first, 
                              **kwargs)
        if Type == 'gru':
            self.net = nn.GRU(input_dim, hidden_dim, 
                              num_layers=num_layers, 
                              bidirectional=bidirectional, 
                              batch_first=batch_first, 
                              **kwargs)
        self.linear = nn.Linear(hidden_dim + bidirectional*hidden_dim, output_dim)


    def forward(self, x):
        lstm_out, _ = self.net(x)
        output = self.linear(lstm_out)
        return output


class Tranpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Tranpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x):
        return torch.transpose(x, self.dim1, self.dim2)

class Drop(nn.Module):
    def __init__(self):
        super(Drop, self).__init__()
    def forward(self, x):
        return x[0]

class ClassicCONV1D(nn.Module):
    """Convolutional network. """
    
    def __init__(self, input_dim=38, hidden_dim1=64, hidden_dim2=128, ks1=3, 
                 ks2=12, output_dim=8):
        super(ClassicCONV1D, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.name = 'ClassicCONV1D'
        self.tranpose = Tranpose(1, 2)
        
        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim1, 
                          kernel_size=ks1, 
                          stride=1, 
                          padding=ks1-1, 
                          dilation=2), 
                nn.LeakyReLU())
        self.conv2 = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dim1, out_channels=hidden_dim2, 
                          kernel_size=ks2, 
                          stride=1, 
                          padding=ks2-1, 
                          dilation=2),
                nn.LeakyReLU())

        self.features = nn.Sequential(
                Tranpose(1, 2),
                nn.Linear(hidden_dim2, hidden_dim1))

        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Sequential(
                nn.Linear(hidden_dim1, output_dim),
                nn.Sigmoid())

    def feature_extract(self, x):
        x = self.tranpose(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.tranpose(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.features(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class MultiCONV1D(nn.Module):
    """MultiConvulational Network."""
    
    def __init__(self, input_dim=38, output_dim=8, out_ch=64):
        super(MultiCONV1D, self).__init__()
        self.name = 'MultiCONV1D'
        self.out_ch = out_ch
        self.hidden_dim1 = self.out_ch//2
        self.tranpose = Tranpose(1, 2)
        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=self.out_ch, 
                          kernel_size=3, 
                          stride=1, 
                          padding=2, 
                          dilation=2),
                nn.LeakyReLU())
        self.conv2 = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, 
                          out_channels=self.out_ch, 
                          kernel_size=7, 
                          stride=1, 
                          padding=6, 
                          dilation=2),
                nn.LeakyReLU())
        self.conv3 = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, 
                          out_channels=self.out_ch, 
                          kernel_size=12, 
                          stride=1, 
                          padding=11, 
                          dilation=2),
                nn.LeakyReLU())
        
        self.conv4 = nn.Sequential(
                nn.Conv1d(in_channels=input_dim,
                          out_channels=self.out_ch, 
                          kernel_size=18, 
                          stride=1, 
                          padding=17, 
                          dilation=2),
                nn.LeakyReLU())

        self.features = nn.Sequential(
                Tranpose(1,2),
                nn.Linear(self.out_ch*4, self.out_ch),
                nn.LeakyReLU(),
                nn.Linear(self.out_ch, self.out_ch//2))
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Sequential(
                nn.Linear(self.out_ch//2, output_dim),
                nn.Sigmoid())

    def feature_extract(self, x):
        x = self.tranpose(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.tranpose(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.features(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class CNN_RNN(nn.Module):
    """ CNN + RNN"""
    
    def __init__(self, conv, hidden_dim = 100, output_dim = 8, num_layers = 3, 
                 batch_first=True, bidirectional=False, type='gru', **kwargs):
        super().__init__(**kwargs)
        input_dim = conv.hidden_dim1
        self.conv = conv
        for param in conv.parameters():
            param.requires_grad = False

        if (type == 'lstm'):
            self.net = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                               bidirectional=bidirectional, 
                               batch_first=batch_first, 
                               **kwargs)
        if (type == 'rnn'):
            self.net = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, 
                              bidirectional=bidirectional, 
                              batch_first=batch_first, 
                              **kwargs)
        if (type == 'gru'):
            self.net = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                              bidirectional=bidirectional, 
                              batch_first=batch_first, 
                              **kwargs)
        self.linear = nn.Linear(hidden_dim+bidirectional*hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv.feature_extract(x)
        out, _ = self.net(x)
        output = self.linear(out)
        return output