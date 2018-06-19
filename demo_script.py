import torch
from utils import calc_mean_IOU
from lib.resume import resume
from lib.validate import validate
from lib.test_minibatch import test_minibatch
import numpy as np
import os
rootDir = os.path.abspath(os.path.dirname(__file__))

### Get a random batch, run it through the network, and save the 3D reconstruction as object files

#################################### PARAMETERS ###########################################
database_path = os.path.join(rootDir, '..', '3D-R2N2', 'ShapeNet')
experiments_path = os.path.join('/home',os.environ['USER'],'experiments','pytorch-3D-R2N2')
experiment_name = 'first_fixed'
epoch = 2
batch_size = 24
num_views = 5
###########################################################################################

import dataset
demo_transform = transforms.Compose([
    transforms.RandomCrop((128, 128)),
    transforms.ToTensor(),
])
if not 'demo_set' in locals():
    print('Reading image info from disk...')
    t1_ImageFolder = time.time()
    demo_set = dataset.Dataset(root=database_path, transform=demo_transform, model_portion=[0.8, 1], min_views=num_views, max_views=num_views, batch_size=batch_size)
    t2_ImageFolder = time.time()
    print('Reading the demo image info took ' + str(round(t2_ImageFolder - t1_ImageFolder, 2)) + ' seconds')
else:
    print('Demo set already loaded')

demo_loader = data.DataLoader(
    dataset=demo_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

import network
encoder = network.Encoder().cuda()
convrnn = network.ConvRNN3d().cuda()
decoder = network.Decoder().cuda()

encoder, convrnn, decoder = resume(encoder, convrnn, decoder, experiments_path, experiment_name, epoch)

mean_test_loss, mean_test_iou = validate(test_loader, encoder, convrnn, decoder)
print('Mean validation loss: ' + str(mean_test_loss))
print('Mean validation IOU: ' + str(mean_test_iou))