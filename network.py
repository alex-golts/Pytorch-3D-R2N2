import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvLSTMCell


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=3)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.LeakyReLU()        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.relu3 = nn.LeakyReLU()        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.relu4 = nn.LeakyReLU()        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2)
        self.relu5 = nn.LeakyReLU()        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool2d(2)
        self.relu6 = nn.LeakyReLU()        
        self.fc1 = nn.Linear(512,1024)
        self.relu7 = nn.LeakyReLU()

    def forward(self, input, hidden1, hidden2, hidden3):
        
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)        
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.pool6(x)
        x = self.relu6(x)
        x = self.fc1(x)
        x = self.relu7(x)
        
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
        
        # The input to the decoder is the hidden state of the 3DConvLSTM, 
        # a tensor of size (B, C, D, H, W), where (D, H, W) are the 3D LSTM grid,
        # (4, 4, 4) in the paper, C is the hidden state size at each point in the grid
        # (128 according to the original code), and B is the batch size.
        
        self.unpool3d1 = nn.MaxUnpool3d((2,2,2))
        self.conv3d1 = nn.Conv3d(256, 128, (3, 3, 3), padding=1)
        self.relu1 = nn.LeakyReLU()
        self.unpool3d2 = nn.MaxUnpool3d((2,2,2))
        self.conv3d2 = nn.Conv3d(128, 128, (3, 3, 3), padding=1)
        self.relu2 = nn.LeakyReLU()
        self.unpool3d3 = nn.MaxUnpool3d((2,2,2))
        self.conv3d3 = nn.Conv3d(128, 64, (3, 3, 3), padding=1)
        self.relu3 = nn.LeakyReLU()
        self.conv3d4 = nn.Conv3d(64, 32, (3, 3, 3), padding=1)        
        self.relu4 = nn.LeakyReLU()
        self.conv3d5 = nn.Conv3d(32, 2, (3, 3, 3), padding=1)
        self.output = nn.Softmax(dim = 2)
        
    def forward(self, input, hidden1, hidden2, hidden3, hidden4):
        
        x = self.unpool3d1(input)
        x = self.conv3d1(x)
        x = self.relu1(x)
        x = self.unpool3d2(x)
        x = self.conv3d2(x)
        x = self.relu2(x)
        x = self.unpool3d3(x)
        x = self.relu3(x)
        x = self.conv3d4(x)
        x = self.relu4(x)
        x = self.conv3d5(x)
        x = self.output(x)
                
        return x
