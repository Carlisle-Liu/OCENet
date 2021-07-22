import torch.nn as nn
import torch
import random
import numpy as np
import torch.nn.functional as F

def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=5):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        a = len(self.losses)
        b = np.maximum(a-self.num, 0)
        c = self.losses[b:]
        #print(c)
        #d = torch.mean(torch.stack(c))
        #print(d)
        return torch.mean(torch.stack(c))

# def save_mask_prediction_example(mask, pred, iter):
# 	plt.imshow(pred[0,:,:],cmap='Greys')
# 	plt.savefig('images/'+str(iter)+"_prediction.png")
# 	plt.imshow(mask[0,:,:],cmap='Greys')
#     plt.savefig('images/'+str(iter)+"_mask.png")

def gradient_x(img):
    # n, c, h, w = img.shape
    # if c == 21:
    #     img = img.view(1, n*c, h, w)
    #     filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    #     filter = torch.tensor(filter, dtype=torch.float32).unsqueeze(0).expand(n*c, 1, 3 ,3)
    #     filter = filter.cuda()
    #     gx = F.conv2d(img, filter, stride=1, padding=1, groups=n*c)
    #     gx = gx.view(n, c, h, w)
    #     return gx
    # else:
    sobel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter = torch.reshape(sobel, [1, 1, 3, 3])
    filter = filter.cuda()
    gx = F.conv2d(img, filter, stride=1, padding=1)
    return gx


def gradient_y(img):
    # n, c, h, w = img.shape
    # if c == 21:
    #     img = img.view(1, n*c, h, w)
    #     filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    #     filter = torch.tensor(filter, dtype=torch.float32).unsqueeze(0).expand(n*c, 1, 3 ,3)
    #     filter = filter.cuda()
    #     gy = F.conv2d(img, filter, stride=1, padding=1, groups=n*c)
    #     gy = gy.view(n, c, h, w)
    #     return gy
    # else:
    sobel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter = torch.reshape(sobel, [1, 1, 3, 3])
    filter = filter.cuda()
    gy = F.conv2d(img, filter, stride=1, padding=1)
    return gy

def normalise(mask):
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask
