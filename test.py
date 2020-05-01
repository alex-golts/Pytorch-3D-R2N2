from lib.resume import resume
from lib.validate import validate
import torch.utils.data as data
from torchvision import transforms
import time
import configargparse


config_file_path = 'config.ini'
parser = configargparse.ArgumentParser(default_config_files=[config_file_path])

parser.add_argument(
    '-c', '--cfg', required=False, is_config_file=True, help='config file path')
parser.add_argument(
    '--database_path', type=str, help='training data location')
parser.add_argument(
    '--saved_models_path', type=str, help='path where to save models')
parser.add_argument(
    '--experiment_name', type=str, help='name of the experiment')    
parser.add_argument(
    '--num_workers', type=int, default=10, help='number of parallel CPU workers')
parser.add_argument(
    '--resume_epoch', type=int, default=0, help='resume from epoch #')
parser.add_argument(
    '--batch_size', type=int, default=8, help='batch size')
parser.add_argument(
    '--min_views', type=int, default=5, help='minimum number of views')
parser.add_argument(
    '--max_views', type=int, default=5, help='maximum number of views')
#parser.add_argument(
#    '--LR', type=float, default=0.0001, help='learning rate')
#parser.add_argument(
#    '--weight_decay', type=float, default=0.00005, help='weight decay')
#parser.add_argument(
#    '--num_epochs', type=int, default=60, help='number of epochs')

    
args = parser.parse_args()

database_path = args.database_path
saved_models_path = args.saved_models_path
experiment_name = args.experiment_name
num_workers = args.num_workers
resume_epoch = args.resume_epoch
batch_size = args.batch_size
min_views = args.min_views
max_views = args.max_views


import dataset
test_transform = transforms.Compose([
    transforms.CenterCrop((128, 128)),
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
    dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

print('total val models: {}; total val batches: {}'.format(
    len(test_set), len(test_loader)))

import network
encoder = network.Encoder().cuda()
convrnn = network.ConvRNN3d().cuda()
decoder = network.Decoder().cuda()

encoder, convrnn, decoder = resume(encoder, convrnn, decoder, saved_models_path, experiment_name, resume_epoch)
mean_test_loss, mean_test_iou = validate(test_loader, encoder, convrnn, decoder)
print('Mean validation loss: ' + str(mean_test_loss))
print('Mean validation IOU: ' + str(mean_test_iou))