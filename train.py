import time
import os
import sys
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as udata
from torchvision import transforms
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils import calc_mean_IOU, Tee
from lib.validate import validate
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
    '--max_views', type=int, default=5, help='maximum number of views')
parser.add_argument(
    '--LR', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--weight_decay', type=float, default=0.00005, help='weight decay')
parser.add_argument(
    '--num_epochs', type=int, default=60, help='number of epochs')

    
args = parser.parse_args()

database_path = args.database_path
saved_models_path = args.saved_models_path
experiment_name = args.experiment_name
num_workers = args.num_workers
resume_epoch = args.resume_epoch
batch_size = args.batch_size
max_views = args.max_views
lr = args.LR
weight_decay = args.weight_decay
num_epochs = args.num_epochs

# save log
if not os.path.isdir(os.path.join(saved_models_path, experiment_name)):
    os.makedirs(os.path.join(saved_models_path, experiment_name))
f = open(os.path.join(saved_models_path, experiment_name, 'log.txt'), 'a')
sys.stdout = Tee(sys.stdout, f)

import dataset

train_transform = transforms.Compose([
    transforms.RandomCrop((128, 128)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.RandomCrop((128, 128)),
    transforms.ToTensor(),
])

if not 'train_set' in locals():
    print('Reading image info from disk...')
    t1_ImageFolder = time.time()
    train_set = dataset.Dataset(root=database_path, transform=train_transform, model_portion=[0, 0.8], min_views=1, max_views=max_views, batch_size=batch_size)
    t2_ImageFolder = time.time()
    print('Reading the train image info took ' + str(round(t2_ImageFolder - t1_ImageFolder, 2)) + ' seconds')
else:
    print('Train set already loaded')

if not 'val_set' in locals():
    print('Reading image info from disk...')
    t1_ImageFolder = time.time()
    val_set = dataset.Dataset(root=database_path, transform=val_transform, model_portion=[0.8, 1], min_views=1, max_views=max_views, batch_size=batch_size)
    t2_ImageFolder = time.time()
    print('Reading the val image info took ' + str(round(t2_ImageFolder - t1_ImageFolder, 2)) + ' seconds')
else:
    print('Val set already loaded')
    
train_loader = udata.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

print('total train models: {}; total train batches: {}'.format(
    len(train_set), len(train_loader)))

val_loader = udata.DataLoader(
    dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

print('total val models: {}; total val batches: {}'.format(
    len(val_set), len(val_loader)))
    
import network

encoder = network.Encoder().cuda()
convrnn = network.ConvRNN3d().cuda()
decoder = network.Decoder().cuda()

NLL = torch.nn.NLLLoss()
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
        os.makedirs(os.path.join(saved_models_path, experiment_name))

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
if resume_epoch > 0:
    resume(resume_epoch)
    last_epoch = resume_epoch
    scheduler.last_epoch = last_epoch - 1

# initialize the hidden state:
hidden = (torch.zeros((batch_size, 128, 4, 4, 4)).cuda(),
               torch.zeros((batch_size, 128, 4, 4, 4)).cuda())


writer = SummaryWriter(logdir=os.path.join(saved_models_path, experiment_name))
# training loop:
it = 0
t1=time.time()
for epoch in range(last_epoch + 1, num_epochs + 1):
    scheduler.step()
    for batch, data in enumerate(train_loader):
        it+=1
        hidden[0].detach_()
        hidden[1].detach_()
        print('Epoch ' + str(epoch) + '/' + str(num_epochs) + ', Batch ' + str(batch) + '/' + str(len(train_loader)))
        solver.zero_grad()
        num_views = data['imgs'].shape[1]
        # loop over the image views and update the 3D-LSTM hidden state
        for v in range(num_views):
            cur_view = torch.squeeze(data['imgs'][:,v,:,:,:]).cuda()
            encoded_vec = encoder(cur_view)
            hidden = convrnn(encoded_vec, hidden)
            if batch%10 == 0:
                x = vutils.make_grid(cur_view, nrow=1, normalize=True, scale_each=True)
                writer.add_image('View ' + str(v), x, it)
        # finally decode the final hidden state and calculate the loss
        output = decoder(hidden[0])
        # torch.exp(output) will return the softmax scores before the log

        loss = NLL(output, data['label'].cuda())
        iou = calc_mean_IOU(torch.exp(output).detach().cpu().numpy(), data['label'].numpy(), 0.5)[5]
        if batch%10 ==0:
            t2=time.time()
            writer.add_scalar('loss', loss, it) 
            writer.add_scalar('IOU', iou, it)
            writer.add_scalar('time per 10 iters', t2-t1, it)
            t1=time.time()
        loss.backward()
        solver.step()
    mean_val_loss, mean_val_iou = validate(val_loader, encoder, convrnn, decoder)
    print('Mean validation loss: ' + str(mean_val_loss))
    print('Mean validation IOU: ' + str(mean_val_iou))
    writer.add_scalar('mean validation loss vs. epoch', mean_val_loss, epoch)
    writer.add_scalar('mean validation IOU vs. epoch', mean_val_iou, epoch)
    save(epoch)




#%%