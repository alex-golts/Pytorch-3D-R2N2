import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvLSTMCell


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # TODO...

    def forward(self, input, hidden1, hidden2, hidden3):
        x = self.conv(input)
        # TODO...

        return x


class 3DConvLSTM(nn.Module):
    def __init__(self):
        super(3DConvLSTM, self).__init__()
        # TODO...

    def forward(self, input):
        x = input
        # TODO...


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(
            32, 512, kernel_size=1, stride=1, padding=0, bias=False)
        

    def forward(self, input, hidden1, hidden2, hidden3, hidden4):
        x = self.conv1(input)
        # TODO...
        
        return x
