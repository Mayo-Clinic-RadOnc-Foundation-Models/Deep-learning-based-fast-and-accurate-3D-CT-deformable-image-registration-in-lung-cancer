import torch
import torch.nn.functional as F
import numpy as np
import math
import pdb
from . import layers#.SpatialTransformer as SpatialTransformer

from ignite.metrics import SSIM as ssim 


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred
        
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        #pdb.set_trace()
        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class MAE:
    """
    Mean absolute error loss.
    """

    def loss (self,y_true,y_pred):
        return torch.mean((torch.abs(y_true-y_pred)))


class soft_MAE_ITV:
    """
    extra loss on tumor
    """
    def loss(self,y_true,y_pred,y_true_str):
#        pdb.set_trace()
        tmp = y_true_str*torch.abs(y_true-y_pred)
        count = torch.sum(y_true_str)
        tmp2 = (y_true.max()-y_true)*(1-y_true_str)*torch.abs(y_true-y_pred) # A*B*C, A: reverse weight, B: other than itv regions, C: mae
        count2 = torch.sum(1-y_true_str)

        return (tmp.sum()/count +tmp2.sum()/count2)

class MAE_ITV:
    """
    extra loss on tumor
    """
    def loss(self,y_true,y_pred,y_true_str):
#        pdb.set_trace()
        tmp = y_true_str*torch.abs(y_true-y_pred)
        count = torch.count_nonzero(y_true_str)
        return tmp.sum()/count


class SSIM:
    """
    structural similarity index
    """
    def loss(self,y_true,y_pred):
        
        metric = ssim(data_range = 1.0)
        str_loss = 0
        for organ in range(y_true.size(2)):
            pr_str = y_pred[:,:,organ,:,:,:]
            tr_str = y_true[:,:,organ,:,:,:]
            pr_str = pr_str.reshape(tr_str.size(0),tr_str.size(1),tr_str.size(2)*tr_str.size(3),tr_str.size(4))
            tr_str = tr_str.reshape(tr_str.size(0),tr_str.size(1),tr_str.size(2)*tr_str.size(3),tr_str.size(4))
            metric.update((pr_str,tr_str))
            str_loss +=metric.compute()
            metric.reset()
        return torch.tensor(1-str_loss/y_true.size(2))

class SSIM2:
    """
    structural similarity index
    """
    def loss(self,y_true,y_pred):
        
        metric = ssim(data_range = 1.0)
        str_loss = 0
        str_loss_all = []
        for organ in range(y_true.size(2)):
            pr_str = y_pred[:,:,organ,:,:,:]
            tr_str = y_true[:,:,organ,:,:,:]
            pr_str = pr_str.reshape(tr_str.size(0),tr_str.size(1),tr_str.size(2)*tr_str.size(3),tr_str.size(4))
            tr_str = tr_str.reshape(tr_str.size(0),tr_str.size(1),tr_str.size(2)*tr_str.size(3),tr_str.size(4))
            metric.update((pr_str,tr_str))
            tmp = metric.compute()
            str_loss +=tmp
            str_loss_all.append(torch.tensor(1-tmp))
            metric.reset()
        return torch.tensor(1-str_loss/y_true.size(2)), torch.tensor(str_loss_all)
        
class wMAE:
    """
    weighted Mean absolute error loss.
    """

    def loss (self,y_true,y_pred):
        tmp = torch.abs(y_true-y_pred)
        tmp = y_true*tmp
        return torch.mean(tmp)

class Dice_update:
    """
    N-D dice for segmentation
    """

#    def loss(self, y_true, y_pred, flow):
    def loss(self, y_true, y_pred):
#        pdb.set_trace()
#        inshape = y_true.shape[1:]
#        sp_trans = layers.SpatialTransformer(inshape)
#        sp_trans = sp_trans.cuda()
#        y_pred = y_pred[:,None,:,:,:]
#        y_true = y_true[:,None,:,:,:]
#        y_pred = sp_trans(y_pred,flow)
#        ndims = len(list(y_pred.size())) - 2
#        vol_axes = list(range(2, ndims + 2))
#        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
#        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        top = 2*(y_true*y_pred).sum()
        bottom = y_true.sum()+y_pred.sum()
        dice = torch.mean(top / bottom)
#        dice = top / bottom
        return -dice

class Dice_update2:
    """
    N-D dice for segmentation
    """

#    def loss(self, y_true, y_pred, flow):
    def loss(self, y_true, y_pred):
#        pdb.set_trace()
#        inshape = y_true.shape[1:]
#        sp_trans = layers.SpatialTransformer(inshape)
#        sp_trans = sp_trans.cuda()
#        y_pred = y_pred[:,None,:,:,:]
#        y_true = y_true[:,None,:,:,:]
#        y_pred = sp_trans(y_pred,flow)
#        ndims = len(list(y_pred.size())) - 2
#        vol_axes = list(range(2, ndims + 2))
#        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
#        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        top = 2*(y_true*y_pred).sum()
        bottom = y_true.sum()+y_pred.sum()
        dice = torch.mean(top / bottom)
#        dice = top / bottom
        dice_o = []
        for organ in range(y_true.size(2)):
            top_o = 2*(y_true[:,:,organ,:,:,:]*y_pred[:,:,organ,:,:,:]).sum()
            bottom_o = y_true[:,:,organ,:,:,:].sum()+y_pred[:,:,organ,:,:,:].sum()
            dice_o.append(-top_o/bottom_o)
        
        return -dice, torch.tensor(dice_o)

class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
