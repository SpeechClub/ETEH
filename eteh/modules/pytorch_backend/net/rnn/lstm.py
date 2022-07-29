"""
Copyright (c) 2019 Microsoft Corporation. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from eteh.utils.data_utils import to_device

class LSTMStack(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super(LSTMStack, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers ,
            batch_first = True,
            dropout = self.dropout,
            bidirectional = self.bidirectional)

    def forward(self, data):
        self.lstm.flatten_parameters()
        output, (h,c) = self.lstm(data)
        
        return output, (h,c)


class RNNCellStack(nn.Module):
    """A pytorch RNNLM

    :param int n_vocab: The size of the vocabulary
    :param int n_layers: The number of layers to create
    :param int n_units: The number of units per layer
    :param str typ: The RNN type
    """

    def __init__(self, input_dim, output_dim, n_layers, n_units, typ="lstm", input_layer="embed"):
        super(RNNCellStack, self).__init__()
        if input_layer == "embed":
            self.embed = nn.Embedding(input_dim, n_units)            
        else:
            self.embed = nn.Linear(input_dim, n_units)
        self.rnn = nn.ModuleList(
            [nn.LSTMCell(n_units, n_units) for _ in range(n_layers)] if typ == "lstm" else [nn.GRUCell(n_units, n_units)
                                                                                            for _ in range(n_layers)])
        self.dropout = nn.ModuleList(
            [nn.Dropout() for _ in range(n_layers + 1)])
        self.lo = nn.Linear(n_units, output_dim)
        self.n_layers = n_layers
        self.n_units = n_units
        self.typ = typ

        # initialize parameters from uniform distribution
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)

    def zero_state(self, batchsize):
        return torch.zeros(batchsize, self.n_units).float()

    def forward(self, state, x):
        import six
        if state is None:
            h = [to_device(self, self.zero_state(x.size(0))) for n in six.moves.range(self.n_layers)]
            state = {'h': h}
            if self.typ == "lstm":
                c = [to_device(self, self.zero_state(x.size(0))) for n in six.moves.range(self.n_layers)]
                state = {'c': c, 'h': h}

        h = [None] * self.n_layers
        emb = self.embed(x)
        if self.typ == "lstm":
            c = [None] * self.n_layers
            h[0], c[0] = self.rnn[0](self.dropout[0](emb), (state['h'][0], state['c'][0]))
            for n in six.moves.range(1, self.n_layers):
                h[n], c[n] = self.rnn[n](self.dropout[n](h[n - 1]), (state['h'][n], state['c'][n]))
            state = {'c': c, 'h': h}
        else:
            h[0] = self.rnn[0](self.dropout[0](emb), state['h'][0])
            for n in six.moves.range(1, self.n_layers):
                h[n] = self.rnn[n](self.dropout[n](h[n - 1]), state['h'][n])
            state = {'h': h}
        y = self.lo(self.dropout[-1](h[-1]))
        return state, y

    def forward_onehot(self, state, x):
        import six
        if state is None:
            h = [to_device(self, self.zero_state(x.size(0))) for n in six.moves.range(self.n_layers)]
            state = {'h': h}
            if self.typ == "lstm":
                c = [to_device(self, self.zero_state(x.size(0))) for n in six.moves.range(self.n_layers)]
                state = {'c': c, 'h': h}
        h = [None] * self.n_layers
        embed_weight = self.embed.weight
        emb = torch.matmul(x, embed_weight)
        if self.typ == "lstm":
            c = [None] * self.n_layers
            h[0], c[0] = self.rnn[0](self.dropout[0](emb), (state['h'][0], state['c'][0]))
            for n in six.moves.range(1, self.n_layers):
                h[n], c[n] = self.rnn[n](self.dropout[n](h[n - 1]), (state['h'][n], state['c'][n]))
            state = {'c': c, 'h': h}
        else:
            h[0] = self.rnn[0](self.dropout[0](emb), state['h'][0])
            for n in six.moves.range(1, self.n_layers):
                h[n] = self.rnn[n](self.dropout[n](h[n - 1]), state['h'][n])
            state = {'h': h}
        y = self.lo(self.dropout[-1](h[-1]))
        return state, y