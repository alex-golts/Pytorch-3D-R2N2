import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

class Conv3dLSTMCell(nn.Module):
    # Implementation of 3D Convolutional LSTM grid according to the 3DR2N2 paper.
    # Input - feature vector (B, C), previous hidden state (B, Nh, N, N, N)
    # Output - (B, Nh, N, N, N)
    def __init__(self,
                 feature_vector_length,
                 hidden_layer_length,
                 grid_size=4,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True):
        super(Conv3dLSTMCell, self).__init__()
        
        self.feature_vector_length = feature_vector_length
        self.hidden_layer_length = hidden_layer_length
        self.grid_size = grid_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        gate_channels = 3 * self.hidden_layer_length
        self.gate_channels = gate_channels
        
        self.linear = nn.Linear(feature_vector_length, 
                                gate_channels*grid_size*grid_size*grid_size)
        self.conv3d = nn.Conv3d(in_channels=hidden_layer_length, 
                                out_channels=gate_channels, 
                                kernel_size=(kernel_size, kernel_size, kernel_size), 
                                stride=1, padding=1, bias=True)  
                                

        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.conv3d.reset_parameters()

    def forward(self, input, hidden):
        
        hx, cx = hidden    
        gates = self.linear(input).view(-1, self.gate_channels, self.grid_size, self.grid_size, self.grid_size) \
                + self.conv3d(hx)
        ingate, forgetgate, cellgate = gates.chunk(3, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate =  torch.tanh(cellgate)
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = torch.tanh(cy)

        return hy, cy
        
        