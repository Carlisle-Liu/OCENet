import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable


def structure_loss(pred, mask, epsilon=1, factor=5):
    weit = 1 + factor * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + epsilon) / (union - inter + epsilon)
    return (wbce + wiou).mean()

def uncertainty_aware_structure_loss(pred, mask, confi_map, f1=0, f2=1, epsilon=1, ep=0):
    # weit = 1 + f1 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) + f2 * confi_map
    weit = 1 + 0 * confi_map
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + epsilon) / (union - inter + epsilon)
    return (wbce + wiou).mean()

def make_confidence_label(pred, gts):
    C_label = (torch.mul(gts, (1 - pred)) + torch.mul((1 - gts), pred))
    return C_label

def make_Dis_label(label, gts):
    D_label = np.ones(gts.shape) * label
    D_label = Variable(torch.FloatTensor(D_label)).cuda()

    return D_label


def compute_energy(disc_score):
    if opt.energy_form == 'tanh':
        energy = torch.tanh(-disc_score.squeeze())
    elif opt.energy_form == 'sigmoid':
        energy = F.sigmoid(-disc_score.squeeze())
    elif opt.energy_form == 'identity':
        energy = -disc_score.squeeze()
    elif opt.energy_form == 'softplus':
        energy = F.softplus(-disc_score.squeeze())
    return energy
