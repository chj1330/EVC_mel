from torch import nn
from modules import Conv1d, Conv1dGLU
from torch.nn import functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self, in_dim=80, preattention=((128, 5, 1),) * 4, convolutions=((128, 5, 1),) * 4, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim

        # Prenet: causal convolution blocks
        self.preattention = nn.ModuleList()
        in_channels = in_dim
        std_mul = 1.0
        for out_channels, kernel_size, dilation in preattention:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.preattention.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.preattention.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.preattention.append(
                Conv1dGLU(in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0

        # Causal convolution blocks + attention layers
        self.convolutions = nn.ModuleList()

        for i, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            assert in_channels == out_channels
            self.convolutions.append(
                Conv1dGLU(in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=False))
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.last_conv = Conv1d(in_channels, in_dim, kernel_size=1,
                                padding=0, dilation=1, std_mul=std_mul,
                                dropout=dropout)

    def forward(self, inputs):
        # Grouping multiple frames if necessary
        assert inputs.size(-1) == self.in_dim

        x = F.dropout(inputs, p=self.dropout, training=self.training)

        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # Prenet
        for f in self.preattention:
            x = f(x)

        # Casual convolutions + Multi-hop attentions
        for f in self.convolutions:
            residual = x
            x = f(x)
            if isinstance(f, Conv1dGLU):
                x = (x + residual) * math.sqrt(0.5)

        x = self.last_conv(x)

        # Back to B x T x C
        x = x.transpose(1, 2)

        # project to mel-spectorgram
        outputs = F.sigmoid(x)
        return outputs

