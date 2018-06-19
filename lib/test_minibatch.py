import torch
from utils import calc_mean_IOU
from utils import voxel2obj
import numpy as np
def test_minibatch(loader, encoder, convrnn, decoder, obj_path):
    NLL = torch.nn.NLLLoss()
    # initialize the hidden state:
    hidden = (torch.zeros((val_loader.batch_size, 128, 4, 4, 4)).cuda(),
              torch.zeros((val_loader.batch_size, 128, 4, 4, 4)).cuda())
    batch, data = next(iter(loader))
    hidden[0].detach_()
    hidden[1].detach_()
    num_views = data['imgs'].shape[1]
    # loop over the image views and update the 3D-LSTM hidden state
    for v in range(num_views):
        cur_view = torch.squeeze(data['imgs'][:,v,:,:,:]).cuda()
        encoded_vec = encoder(cur_view)
        hidden = convrnn(encoded_vec, hidden)
    # finally decode the final hidden state and calculate the loss
    output = decoder(hidden[0])
    output_prob = torch.exp(output)   
    voxel2obj(obj_path, output_prob[0, 1, :, :, :] > 0.5)
    # TODO: this was originally output_prob[0, :, 1, :, :]. check what this implies!
    # TODO: also write the original images to disk
    loss = NLL(output, data['label'].cuda()).item()
    iou = calc_mean_IOU(output_prob.detach().cpu().numpy(), data['label'].numpy(), 0.5)[5]
    print('Loss=' + str(loss) + ', IOU=' + str(iou))
    return loss, iou