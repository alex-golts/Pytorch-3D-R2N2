import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.modules.utils import _pair


class ConvRNNCellBase(nn.Module):
    def __repr__(self):
        s = (
            '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'
            ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.hidden_kernel_size = _pair(hidden_kernel_size)

        hidden_padding = _pair(hidden_kernel_size // 2)

        gate_channels = 4 * self.hidden_channels
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.conv_ih(input) + self.conv_hh(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy
        
        
        


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

        
        #gate_channels = 4 * self.hidden_channels
        
        self.linear = nn.Linear(feature_vector_length, hidden_layer_length*grid_size*grid_size*grid_size)
        self.conv3d = nn.Conv3d(in_channels=hidden_layer_length, out_channels=hidden_layer_length, kernel_size=(kernel_size, kernel_size, kernel_size), stride=1, padding=1, bias=True)       
        
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.conv_ih(input) + self.conv_hh(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy
        
        
        



class FCConv3DLayer(Layer):
    """3D Convolution layer with FC input and hidden unit"""

    def __init__(self, prev_layer, fc_layer, filter_shape, padding=None, params=None):
        """Prev layer is the 3D hidden layer"""
        
        # prev_layer: (B,Nh,N,N,N)
        # fc_layer: (B,C)
        # filter_shape: (Nh, Nh, 3, 3, 3). (out channel, in channel, time, height, width)
        super().__init__(prev_layer)
        self._fc_layer = fc_layer
        self._filter_shape = [filter_shape[0],  # out channel
                              filter_shape[2],  # time
                              filter_shape[1],  # in channel
                              filter_shape[3],  # height
                              filter_shape[4]]  # width
        self._padding = padding

        if padding is None:
            self._padding = [0, int((self._filter_shape[1] - 1) / 2), 0, int(
                (self._filter_shape[3] - 1) / 2), int((self._filter_shape[4] - 1) / 2)]


        self._output_shape = [self._input_shape[0], self._input_shape[1], filter_shape[0],
                              self._input_shape[3], self._input_shape[4]]
                              # self._input_shape = prev_layer.output_shape (B,Nh,Nh,N,N)

        if params is None:
            self.Wh = Weight(self._filter_shape, is_bias=False) # (Nh,Nh,3,3,3)

            self._Wx_shape = [self._fc_layer._output_shape[1], np.prod(self._output_shape[1:])] # (1024, Nh*Nh*N*N)

            # Each 3D cell will have independent weights but for computational
            # speed, we expand the cells and compute a matrix multiplication.
            self.Wx = Weight(
                self._Wx_shape,
                is_bias=False,
                fan_in=self._input_shape[1],
                fan_out=self._output_shape[2])

            self.b = Weight((filter_shape[0],), is_bias=True, mean=0.1, filler='constant')
            params = [self.Wh, self.Wx, self.b]
        else:
            self.Wh = params[0]
            self.Wx = params[1]
            self.b = params[2]

        self.params = [self.Wh, self.Wx, self.b]

    def set_output(self):
        padding = self._padding
        input_shape = self._input_shape
        padded_input = tensor.alloc(0.0,  # Value to fill the tensor
                                    input_shape[0],
                                    input_shape[1] + 2 * padding[1],
                                    input_shape[2],
                                    input_shape[3] + 2 * padding[3],
                                    input_shape[4] + 2 * padding[4])

        padded_input = tensor.set_subtensor(padded_input[:, padding[1]:padding[1] + input_shape[
            1], :, padding[3]:padding[3] + input_shape[3], padding[4]:padding[4] + input_shape[4]],
                                            self._prev_layer.output)

        fc_output = tensor.reshape(
            tensor.dot(self._fc_layer.output, self.Wx.val), self._output_shape)
        self._output = conv3d2d.conv3d(padded_input, self.Wh.val) + \
            fc_output + self.b.val.dimshuffle('x', 'x', 0, 'x', 'x')







        
#
#class Conv3DLSTMLayer(Layer):
#    """Convolution 3D LSTM layer
#
#    Unlike a standard LSTM cell witch doesn't have a spatial information,
#    Convolutional 3D LSTM has limited connection that respects spatial
#    configuration of LSTM cells.
#
#    The filter_shape defines the size of neighbor that the 3D LSTM cells will consider.
#    """
#
#    def __init__(self, prev_layer, filter_shape, padding=None, params=None):
#
#        super().__init__(prev_layer)
#        prev_layer._input_shape
#        n_c = filter_shape[0] 
#        n_x = self._input_shape[2] # length of input feature vector - 1024
#        n_neighbor_d = filter_shape[1] # 3
#        n_neighbor_h = filter_shape[2] # 3
#        n_neighbor_w = filter_shape[3] # 3
#
#        # Compute all gates in one convolution
#        self._gate_filter_shape = [4 * n_c, 1, n_x + n_c, 1, 1]
#
#        self._filter_shape = [filter_shape[0],  # num out hidden representation
#                              filter_shape[1],  # time
#                              self._input_shape[2],  # in channel
#                              filter_shape[2],  # height
#                              filter_shape[3]]  # width
#        self._padding = padding
#
#        # signals: (batch,       in channel, depth_i, row_i, column_i)
#        # filters: (out channel, in channel, depth_f, row_f, column_f)
#
#        # there are "num input feature maps * filter height * filter width"
#        # inputs to each hidden unit
#        if params is None:
#            self.W = Weight(self._filter_shape, is_bias=False)
#            self.b = Weight((filter_shape[0],), is_bias=True, mean=0.1, filler='constant')
#            params = [self.W, self.b]
#        else:
#            self.W = params[0]
#            self.b = params[1]
#
#        self.params = [self.W, self.b]
#
#        if padding is None:
#            self._padding = [0, int((filter_shape[1] - 1) / 2), 0, int((filter_shape[2] - 1) / 2),
#                             int((filter_shape[3] - 1) / 2)]
#
#        self._output_shape = [self._input_shape[0], self._input_shape[1], filter_shape[0],
#                              self._input_shape[3], self._input_shape[4]]
#
#    def set_output(self):
#        padding = self._padding
#        input_shape = self._input_shape
#        padded_input = tensor.alloc(0.0,  # Value to fill the tensor
#                                    input_shape[0],
#                                    input_shape[1] + 2 * padding[1],
#                                    input_shape[2],
#                                    input_shape[3] + 2 * padding[3],
#                                    input_shape[4] + 2 * padding[4])
#
#        padded_input = tensor.set_subtensor(padded_input[:, padding[1]:padding[1] + input_shape[
#            1], :, padding[3]:padding[3] + input_shape[3], padding[4]:padding[4] + input_shape[4]],
#                                            self._prev_layer.output)
#
#        self._output = conv3d2d.conv3d(padded_input, self.W.val) + \
#            self.b.val.dimshuffle('x', 'x', 0, 'x', 'x')
