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
convrnn = network.ConvRNN3d().cuda()
decoder = network.Decoder().cuda()

# dummy test:
dummy_input = torch.randn((16, 3, 128, 128)).cuda() # In older than 0.4 pytorch, this would need to be wrapped by 'Variable'
import pdb
pdb.set_trace()
encoded_vec = encoder(dummy_input)
hidden_state = convrnn(encoded_vec)
output = decoder(hidden_state)

