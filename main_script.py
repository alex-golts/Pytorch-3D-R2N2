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

database_path = os.path.join(rootDir, '..', '3D-R2N2', 'ShapeNet')
random_seed = 0 # None for randomized seed
model_name = '3d-lstm-3'
saved_models_path = os.path.join('/home',os.environ['USER'],'experiments','pytorch-3D-R2N2')
experiment_name = 'debug'
resume_epoch = None
batch_size = 24
iters = 60000
weights = None # for initialization
max_views = 5
lr = 0.0005
max_epochs = 40

### END PARAMETERS ###


import dataset

train_transform = transforms.Compose([
    transforms.RandomCrop((128, 128)),
    transforms.ToTensor(),
])

t1_ImageFolder = time.time()

train_set = dataset.Dataset(root=os.path.join(database_path, 'ShapeNetRendering'), transform=train_transform, model_portion=[0, 0.8], max_views=max_views, batch_size=batch_size)
t2_ImageFolder = time.time()
print('Reading the image folder took ' + str(round(t2_ImageFolder - t1_ImageFolder, 2)) + ' seconds')

train_loader = data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

print('total models: {}; total batches: {}'.format(
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


solver = optim.Adam(
    [
        {
            'params': encoder.parameters()
        },
        {
            'params': convrnn.parameters()
        },
        {
            'params': decoder.parameters()
        },
    ],
    lr=lr)


def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    encoder.load_state_dict(
        torch.load(os.path.join(saved_models_path, experiment_name, 'encoder_{}_{:08d}.pth'.format(s, epoch))))
    convrnn.load_state_dict(
        torch.load(os.path.join(saved_models_path, experiment_name, 'convrnn_{}_{:08d}.pth'.format(s, epoch))))
    decoder.load_state_dict(
        torch.load(os.path.join(saved_models_path, experiment_name, 'decoder_{}_{:08d}.pth'.format(s, epoch))))



def save(index, epoch=True):
    if not os.path.exists(os.path.join(saved_models_path, experiment_name)):
        os.mkdir(os.path.join(saved_models_path, experiment_name))

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(encoder.state_dict(), os.path.join(saved_models_path, experiment_name, 'encoder_{}_{:08d}.pth'.format(
        s, index)))

    torch.save(convrnn.state_dict(),
               os.path.join(saved_models_path, experiment_name, 'convrnn_{}_{:08d}.pth'.format(s, index)))

    torch.save(decoder.state_dict(), os.path.join(saved_models_path, experiment_name, 'decoder_{}_{:08d}.pth'.format(
        s, index)))
        
# decay the learning rate by gamma when reaching 3,10,20,... epochs
scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)

last_epoch = 0
if resume_epoch != None:
    resume(resume_epoch)
    last_epoch = resume_epoch
    scheduler.last_epoch = last_epoch - 1

for epoch in range(last_epoch + 1, max_epochs + 1):

    scheduler.step()
    
    for batch, data in enumerate(train_loader):
        print (str(data.shape))
    
    save(epoch)




