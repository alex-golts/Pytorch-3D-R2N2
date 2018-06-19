import torch
from utils import calc_mean_IOU
from lib.resume import resume
from lib.validate import validate
import numpy as np

rootDir = os.path.abspath(os.path.dirname(__file__))

#################################### PARAMETERS ###########################################
database_path = os.path.join(rootDir, '..', '3D-R2N2', 'ShapeNet')
experiments_path = os.path.join('/home',os.environ['USER'],'experiments','pytorch-3D-R2N2')
experiment_name = 'first_fixed'
epoch = 2
batch_size = 24
min_views = 5
max_views = 5
###########################################################################################

import dataset
test_transform = transforms.Compose([
    transforms.RandomCrop((128, 128)),
    transforms.ToTensor(),
])
if not 'test_set' in locals():
    print('Reading image info from disk...')
    t1_ImageFolder = time.time()
    test_set = dataset.Dataset(root=database_path, transform=test_transform, model_portion=[0.8, 1], min_views=min_views, max_views=max_views, batch_size=batch_size)
    t2_ImageFolder = time.time()
    print('Reading the test image info took ' + str(round(t2_ImageFolder - t1_ImageFolder, 2)) + ' seconds')
else:
    print('Test set already loaded')

test_loader = data.DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

print('total val models: {}; total val batches: {}'.format(
    len(test_set), len(test_loader)))

import network
encoder = network.Encoder().cuda()
convrnn = network.ConvRNN3d().cuda()
decoder = network.Decoder().cuda()

encoder, convrnn, decoder = resume(encoder, convrnn, decoder, experiments_path, experiment_name, epoch)
mean_test_loss, mean_test_iou = validate(test_loader, encoder, convrnn, decoder)
print('Mean validation loss: ' + str(mean_test_loss))
print('Mean validation IOU: ' + str(mean_test_iou))