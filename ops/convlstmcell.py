# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

import torch
import torch.nn as nn
import torch.nn.modules.utils as utils


## Convolutional LSTM Module
class ConvLSTMCell(nn.Module):

    def __init__(self,input_channels, hidden_channels,
                 order=3, steps=3, ranks=8,
                 kernel_size=5, bias=True):

        super(ConvLSTMCell, self).__init__()

        ## Input/output interfaces
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.steps = steps
        self.order = order
        self.lags = steps - order + 1
        kernel_size = utils._pair(kernel_size)
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        Conv2d = lambda in_channels, out_channels: nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)

        Conv3d = lambda in_channels, out_channels: nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, bias=bias,
            kernel_size=kernel_size + (self.lags,), padding=padding + (0,))

        ## Convolutional layers
        self.layers = nn.ModuleList()
        self.layers_ = nn.ModuleList()
        for l in range(order - 1):
            self.layers.append(Conv2d(
                in_channels=ranks if l < order - 1 else ranks,
                out_channels=ranks if l < order - 1 else 1 * hidden_channels))
        for l in range(order):
            self.layers_.append(Conv3d(
                in_channels=hidden_channels, out_channels=ranks))

        self.LSTM_W_hi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        self.LSTM_W_hf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        self.LSTM_W_hg = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        self.LSTM_W_ho = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)

        self.LSTM_U_pi = nn.Conv2d(ranks, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.LSTM_U_pf = nn.Conv2d(ranks, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.LSTM_U_pg = nn.Conv2d(ranks, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.LSTM_U_po = nn.Conv2d(ranks, hidden_channels, kernel_size=3, padding=1, bias=False)

    def initialize(self, inputs):
        device = inputs.device
        batch_size, _, height, width = inputs.size()

        self.hidden_states = [torch.zeros(batch_size, self.hidden_channels,
                                          height, width, device=device) for t in range(self.steps)]
        self.hidden_pointer = 0

        self.cell_states = torch.zeros(batch_size,
                                       self.hidden_channels, height, width, device=device)

    def forward(self, inputs, first_step=False):

        if first_step: self.initialize(inputs)

        # (1) Accumulate Temporary State (ATS) Module
        for l in range(self.order):
            input_pointer = self.hidden_pointer if l == 0 else (input_pointer + 1) % self.steps

            input_states = self.hidden_states[input_pointer:] + self.hidden_states[:input_pointer]
            input_states = input_states[:self.lags]

            input_states = torch.stack(input_states, dim=-1)
            input_states = self.layers_[l](input_states)
            input_states = torch.squeeze(input_states, dim=-1)

            if l == 0:
                temp_states = input_states
            else:
                temp_states = input_states + self.layers[l - 1](temp_states)

        # (2) Standard convolutional-LSTM module
        Uh_i = self.LSTM_U_pi(temp_states)
        Uh_f = self.LSTM_U_pf(temp_states)
        Uh_o = self.LSTM_U_po(temp_states)
        Uh_g = self.LSTM_U_pg(temp_states)
        Wh_i = self.LSTM_W_hi(inputs)
        Wh_f = self.LSTM_W_hf(inputs)
        Wh_o = self.LSTM_W_ho(inputs)
        Wh_g = self.LSTM_W_hg(inputs)
        i = torch.sigmoid(Wh_i + Uh_i)
        f = torch.sigmoid(Wh_f + Uh_f)
        o = torch.sigmoid(Wh_o + Uh_o)
        g = torch.tanh(Wh_g + Uh_g)

        self.cell_states = f * self.cell_states + i * g
        outputs = o * torch.tanh(self.cell_states)
        self.hidden_states[self.hidden_pointer] = outputs
        self.hidden_pointer = (self.hidden_pointer + 1) % self.steps

        return outputs
