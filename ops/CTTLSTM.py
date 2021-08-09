import torch
from torch import nn
from ops.convlstmcell import ConvLSTMCell
from opts import parser

args = parser.parse_args()


class _convlstmBlock(nn.Module):
    def __init__(self, in_channels):
        super(_convlstmBlock, self).__init__()
        self.in_channels = in_channels
        self.output_channels = in_channels // 4

        if in_channels == 1024:
            self.order = 5
            self.steps = 5
            self.ranks = 128
            self.kernel_size = 1

        if in_channels == 256:
            self.order = 5
            self.steps = 5
            self.ranks = 128
            self.kernel_size = 1

            self.output_channels = in_channels // 2

        if in_channels == 512:
            self.order = 5
            self.steps = 5
            self.ranks = 128
            self.kernel_size = 1

        if in_channels == 2048:
            self.order = 5
            self.steps = 5
            self.ranks = 128
            self.kernel_size = 1

        if in_channels == 256:
            self.reduce_dim = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True))

            self.cttlstm = ConvLSTMCell(
                input_channels=self.output_channels, hidden_channels=self.output_channels,
                order=self.order, steps=self.steps, ranks=self.ranks,
                kernel_size=self.kernel_size, bias=False)
            self.up_dim = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True))
            self.short_cut = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)

        if in_channels == 512:
            self.reduce_dim = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True))

            self.cttlstm = ConvLSTMCell(
                input_channels=self.output_channels, hidden_channels=self.output_channels,
                order=self.order, steps=self.steps, ranks=self.ranks,
                kernel_size=self.kernel_size, bias=False)
            self.up_dim = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True))
            self.short_cut = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)

        if in_channels == 1024:
            self.reduce_dim = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True))

            self.cttlstm = ConvLSTMCell(
                input_channels=self.output_channels, hidden_channels=self.output_channels,
                order=self.order, steps=self.steps, ranks=self.ranks,
                kernel_size=self.kernel_size, bias=False)
            self.up_dim = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True))
            self.short_cut = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)

        if in_channels == 2048:
            self.reduce_dim = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True))

            self.cttlstm = ConvLSTMCell(
                input_channels=self.output_channels, hidden_channels=self.output_channels,
                order=self.order, steps=self.steps, ranks=self.ranks,
                kernel_size=self.kernel_size, bias=False)
            self.up_dim = nn.Sequential(nn.Conv2d(in_channels // 4, in_channels, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True))
            self.short_cut = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, inputs):
        if self.in_channels == 256:
            self.in_shape = inputs.size()
            self.timesteps = args.num_segments
            self.batch_size = inputs.size(0) // self.timesteps
            if self.batch_size == 0:
                self.batch_size = 1
            res = self.short_cut(inputs)
            inputs = self.reduce_dim(inputs)
            inputs = inputs.view(self.batch_size, self.timesteps, self.in_shape[1] // 2, self.in_shape[2],
                                 self.in_shape[3])

            outputs = [None] * self.timesteps

            for t in range(self.timesteps):
                inputs_ = inputs[:, t, :, :, :]
                first_step = (t == 0)
                outputs[t] = self.cttlstm(inputs_, first_step)
            assignments = torch.stack(outputs, dim=0)
            assignments = torch.transpose(assignments, 0, 1).contiguous()
            cur_batch_size = assignments.shape[0]
            assignments = assignments.view(cur_batch_size * self.timesteps, self.in_channels // 2, self.in_shape[2],
                                           self.in_shape[3])

            final = assignments
            final = self.up_dim(final)
            return final + res

        if self.in_channels == 512:
            self.in_shape = inputs.size()
            self.timesteps = args.num_segments
            self.batch_size = inputs.size(0) // self.timesteps
            if self.batch_size == 0:
                self.batch_size = 1
            res = self.short_cut(inputs)
            inputs = self.reduce_dim(inputs)
            inputs = inputs.view(self.batch_size, self.timesteps, self.in_shape[1] // 4, self.in_shape[2],
                                 self.in_shape[3])

            outputs = [None] * self.timesteps

            for t in range(self.timesteps):
                inputs_ = inputs[:, t, :, :, :]
                first_step = (t == 0)
                outputs[t] = self.cttlstm(inputs_, first_step)
            assignments = torch.stack(outputs, dim=0)
            assignments = torch.transpose(assignments, 0, 1).contiguous()
            cur_batch_size = assignments.shape[0]
            assignments = assignments.view(cur_batch_size * self.timesteps, self.in_channels // 4, self.in_shape[2],
                                           self.in_shape[3])
            final = assignments
            final = self.up_dim(final)
            return final + res

        if self.in_channels == 1024:
            self.in_shape = inputs.size()
            self.timesteps = args.num_segments
            self.batch_size = inputs.size(0) // self.timesteps
            if self.batch_size == 0:
                self.batch_size = 1
            res = self.short_cut(inputs)
            inputs = self.reduce_dim(inputs)
            inputs = inputs.view(self.batch_size, self.timesteps, self.in_shape[1] // 4, self.in_shape[2],
                                 self.in_shape[3])

            outputs = [None] * self.timesteps

            for t in range(self.timesteps):
                inputs_ = inputs[:, t, :, :, :]
                first_step = (t == 0)
                outputs[t] = self.cttlstm(inputs_, first_step)
            assignments = torch.stack(outputs, dim=0)
            assignments = torch.transpose(assignments, 0, 1).contiguous()
            cur_batch_size = assignments.shape[0]
            assignments = assignments.view(cur_batch_size * self.timesteps, self.in_channels // 4, self.in_shape[2],
                                           self.in_shape[3])
            final = assignments
            final = self.up_dim(final)
            return final + res

        if self.in_channels == 2048:
            self.in_shape = inputs.size()
            self.timesteps = args.num_segments
            self.batch_size = inputs.size(0) // self.timesteps
            if self.batch_size == 0:
                self.batch_size = 1
            res = self.short_cut(inputs)

            inputs = self.reduce_dim(inputs)
            inputs = inputs.view(self.batch_size, self.timesteps, self.in_shape[1] // 4, self.in_shape[2],
                                 self.in_shape[3])
            outputs = [None] * self.timesteps

            for t in range(self.timesteps):
                inputs_ = inputs[:, t, :, :, :]
                first_step = (t == 0)
                outputs[t] = self.cttlstm(inputs_, first_step)
            assignments = torch.stack(outputs, dim=0)
            assignments = torch.transpose(assignments, 0, 1).contiguous()
            cur_batch_size = assignments.shape[0]
            assignments = assignments.view(cur_batch_size * self.timesteps, self.in_channels // 4, self.in_shape[2],
                                           self.in_shape[3])

            final = assignments
            final = self.up_dim(final)
            return final + res
