import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable


def structure_loss(pred, mask, epsilon=1, factor=5):
    """
    :param pred: COD prediction
    :param mask: COD GT
    :param epsilon: a small number preventing division by 0 in computing IoU loss
    :param factor: hyperparameter
    :return: structure loss value
    """
    weit = 1 + factor * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + epsilon) / (union - inter + epsilon)
    return (wbce + wiou).mean()

def uncertainty_aware_structure_loss(pred, mask, confi_map, epoch, f1=1, f2=10, epsilon=1):
    """
    :param pred: COD prediction
    :param mask: COD GT
    :param confi_map: OCE_Net prediction
    :param epoch: training epoch
    :param f1: hyperparameter
    :param f2: hyperparameter
    :param epsilon: a small number preventing division by 0 in computing IoU loss
    :return: loss value
    """
    if epoch < 20:
        f2 = 0
    weit = 1 + f1 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) + f2 * confi_map
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + epsilon) / (union - inter + epsilon)
    return (wbce + wiou).mean()

def make_confidence_label(pred, gts):
    """
    :param pred: COD prediction
    :param gts: COD GT
    :return: OCE_Net supervision
    """
    C_label = (torch.mul(gts, (1 - pred)) + torch.mul((1 - gts), pred))
    return C_label
