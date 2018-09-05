import torch
import os
def resume(encoder, convrnn, decoder, experiments_path, experiment_name, epoch):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    encoder.load_state_dict(
        torch.load(os.path.join(experiments_path, experiment_name, 'encoder_{}_{:08d}.pth'.format(s, epoch))))
    convrnn.load_state_dict(
        torch.load(os.path.join(experiments_path, experiment_name, 'convrnn_{}_{:08d}.pth'.format(s, epoch))))
    decoder.load_state_dict(
        torch.load(os.path.join(experiments_path, experiment_name, 'decoder_{}_{:08d}.pth'.format(s, epoch))))

    return encoder, convrnn, decoder