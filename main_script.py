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
database_path = os.path.join(rootDir, '..', '3D-R2N2', 'ShapeNet')
experiment_name = 'debug'
batch_size = 24
iters = 60000
weights = None # for initialization
max_view = 5


### END PARAMETERS ###


import dataset

train_transform = transforms.Compose([
    transforms.RandomCrop((128, 128)),
    transforms.ToTensor(),
])
#import pdb
#pdb.set_trace()

t1_ImageFolder = time.time()

train_set = dataset.Dataset(root=os.path.join(database_path, 'ShapeNetRendering'), transform=train_transform, model_portion=[0, 0.8])
t2_ImageFolder = time.time()
print('Reading the image folder took ' + str(round(t2_ImageFolder - t1_ImageFolder, 2)) + ' seconds')

import pdb
pdb.set_trace()
train_loader = data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1)

print('total images: {}; total batches: {}'.format(
    len(train_set), len(train_loader)))

import network

encoder = network.Encoder().cuda()
convrnn = network.ConvRNN3d().cuda()
decoder = network.Decoder().cuda()

if False:
    # dummy test:
    dummy_input = torch.randn((16, 3, 128, 128)).cuda() # In older than 0.4 pytorch, this would need to be wrapped by 'Variable'
    encoded_vec = encoder(dummy_input)
    hidden0 = (torch.zeros((16, 128, 4, 4, 4)).cuda(),
               torch.zeros((16, 128, 4, 4, 4)).cuda())
    hidden = convrnn(encoded_vec, hidden0)
    output = decoder(hidden[0])



