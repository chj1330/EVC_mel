# coding: utf-8

import torch
from torch import nn
import math
import numpy as np
from torch.nn import functional as F


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, std_mul=4.0, **kwargs):
    from conv import Conv1d
    m = Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class Conv1dGLU(nn.Module):
    """(Dilated) Conv1d + Gated linear unit + (optionally) speaker embedding
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 dropout, padding=None, dilation=1, causal=False, residual=False,
                 *args, **kwargs):
        super(Conv1dGLU, self).__init__()
        self.dropout = dropout
        self.residual = residual
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        self.conv = Conv1d(in_channels, 2 * out_channels, kernel_size,
                           dropout=dropout, padding=padding, dilation=dilation,
                           *args, **kwargs)

    def forward(self, x):
        return self._forward(x, False)

    def incremental_forward(self, x):
        return self._forward(x, True)

    def _forward(self, x, is_incremental):
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)

        x = a * F.sigmoid(b)
        return (x + residual) * math.sqrt(0.5) if self.residual else x

    def clear_buffer(self):
        self.conv.clear_buffer()



