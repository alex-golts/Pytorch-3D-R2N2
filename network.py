import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvLSTMCell


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(512,1024)

    def forward(self, input, hidden1, hidden2, hidden3):
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = self.pool6(x)
        
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
