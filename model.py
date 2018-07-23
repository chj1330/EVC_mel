# coding: utf-8

from torch import nn
from modules import Conv1d, Conv1dGLU
from torch.nn import functional as F
import math

class EVCModel(nn.Module):
    """Attention seq2seq model + post processing network
    """
    def __init__(self, NEU2EMO, MEL2LIN, mel_dim=80, linear_dim=513):
        super(EVCModel, self).__init__()
        self.seq2seq = NEU2EMO
        self.postnet = MEL2LIN  # referred as "Converter" in DeepVoice3
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim

    def make_generation_fast_(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(remove_weight_norm)


    def forward(self, S_MEL, T_MEL=None):
        # Apply seq2seq
        # (B, T//r, mel_dim*r)
        mel_outputs = self.seq2seq(S_MEL)

        # Prepare postnet inputs
        if self.seq2seq == None:
            postnet_inputs = T_MEL
        else:
            postnet_inputs = mel_outputs

        # (B, T, linear_dim)
        # Convert coarse mel-spectrogram (or decoder hidden states) to
        # high resolution spectrogram
        linear_outputs = self.postnet(postnet_inputs)
        assert linear_outputs.size(-1) == self.linear_dim

        return mel_outputs, linear_outputs



class NEU2EMO(nn.Module):
    def __init__(self, in_dim=80, preattention=((128, 5, 1),) * 4, convolutions=((128, 5, 1),) * 4, dropout=0.1):
        super(NEU2EMO, self).__init__()
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



class MEL2LIN(nn.Module):
    def __init__(self, in_dim=80, out_dim=513, convolutions=((256, 5, 1),) * 4, dropout=0.1):
        super(MEL2LIN, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Non causual convolution blocks
        in_channels = convolutions[0][0]

        self.convolutions = nn.ModuleList([
            # 1x1 convolution first
            Conv1d(in_dim, in_channels, kernel_size=1, padding=0, dilation=1,
                   std_mul=1.0),
            Conv1dGLU(in_channels, in_channels, kernel_size=3, causal=False,
                      dilation=3, dropout=dropout, std_mul=4.0, residual=True),
        ])

        std_mul = 4.0
        for (out_channels, kernel_size, dilation) in convolutions:
            if in_channels != out_channels:
                self.convolutions.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1dGLU(in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.convolutions.append(Conv1d(in_channels, out_dim, kernel_size=1,
                                        padding=0, dilation=1, std_mul=std_mul,
                                        dropout=dropout))

    def forward(self, x):
        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)

        for f in self.convolutions:
            x = f(x)
        # Back to B x T x C
        x = x.transpose(1, 2)

        return F.sigmoid(x)