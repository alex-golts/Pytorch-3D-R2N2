import time
import os
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms

rootDir = os.path.abspath(os.path.dirname(__file__))

### PARAMETERS ###

random_seed = 0 # None for randomized seed
model_name = '3d-lstm-3'
saved_models_path = os.path.join('/home',os.environ['USER'],'experiments','pytorch-3D-R2N2')
experiment_name = 'debug'
batch_size = 24
iters = 60000
weights = None # for initialization

### END PARAMETERS ###

import network

encoder = network.Encoder().cuda()
binarizer = network.3DConvLSTM().cuda()
decoder = network.Decoder().cuda()
