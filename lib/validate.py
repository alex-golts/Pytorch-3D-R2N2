import torch
from utils import calc_mean_IOU
import numpy as np
def validate(val_loader, encoder, convrnn, decoder):
    print('Validating...')
    NLL = torch.nn.NLLLoss()
    # initialize the hidden state:
    losses = []
    ious = []
    hidden = (torch.zeros((val_loader.batch_size, 128, 4, 4, 4)).cuda(),
              torch.zeros((val_loader.batch_size, 128, 4, 4, 4)).cuda())
    for batch, data in enumerate(val_loader):
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
        # torch.exp(output) will return the softmax scores before the log
        loss = NLL(output, data['label'].cuda()).item()
        iou = calc_mean_IOU(torch.exp(output).detach().cpu().numpy(), data['label'].numpy(), 0.4)[5]
        print('Batch ' + str(batch) + '/' + str(len(val_loader)) + ': Loss=' + str(loss) + ', IOU=' + str(iou))
        losses.append(loss)
        ious.append(iou)
    return np.mean(losses), np.mean(ious)