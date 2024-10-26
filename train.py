#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# 1.use the same scale to normalize data. 
# 2.use weighted MAE as the loss
# 3.use mask. randomly crop out  5*5*5 region each time   
# 4. Use SSIM loss
# 5. without BODY contour, add itv_ctv
import os
import random
import argparse
import time
import numpy as np
import torch

from ignite.metrics import SSIM
import pdb
import gc
gc.collect()

import h5py


# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', default = './data_pre/train_lung_index_v7.txt', help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models10',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--organ-names',default = ['itv_ctv','lung_r','lung_l','esophagus','heart','cord'],help = 'bitmap organ name')
parser.add_argument('--mks', type = int, default = 5, help= 'mask size')
parser.add_argument('--gpu', default='0,1,2,3', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=4, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=50,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', default = './models9/0665.pt',help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='wmae',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir

# load and prepare training data

def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist

train_files = read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)

assert len(train_files) > 0, 'Could not find any training data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel
generator = vxm.generators.scan_to_scan_ctlung_withbitmap3(
        train_files,organ_names = args.organ_names, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)


# extract shape from sampled input
#pdb.set_trace()
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
#pdb.set_trace()
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    model = vxm.networks.VxmDense_bitmap.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.VxmDense_bitmap(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

# pdb.set_trace()
if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
#if args.image_loss =='wmae_dice':
#    losses = [vxm.losses.wMAE().loss,vxm.losses.Dice().loss]
#    weights = [1,0.5]
if args.image_loss =='wmae':
    losses = [vxm.losses.wMAE().loss]
    weights = [1]
elif args.image_loss == 'ncc':
    losses = [vxm.losses.NCC().loss]
    weights = [1]
elif args.image_loss == 'mse':
    losses = [vxm.losses.MSE().loss]
    weights = [1]
elif args.image_loss =='mse_ncc':
    losses = [vxm.losses.MSE().loss,vxm.losses.NCC().loss]
    weights = [1,0.1]
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

str_loss = vxm.losses.SSIM().loss

# need two image loss functions if bidirectional
#if bidir:
#    losses = [image_loss_func, image_loss_func]
#    weights = [0.5, 0.5]
#else:
#    losses = [image_loss_func]
#    weights = [1]

# prepare deformation loss
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

# training loops

best_mae_loss = 10000
best_ssim_loss = 10000
for epoch in range(args.initial_epoch, args.epochs):


    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    
    
    # save model checkpoint
    if epoch  == 0:
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))
    
    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true,inputs_str,y_true_str = next(generator)
        inputs = inputs.squeeze(-1)
        y_true = y_true.squeeze(-1)
        mask = torch.ones(inputs.size())
        x_index = torch.randint(0, mask.size(2)-args.mks,(mask.size(0),))
        y_index = torch.randint(10, 310,(mask.size(0),))
        z_index = torch.randint(0, mask.size(4)-args.mks,(mask.size(0),))
        for i in range(mask.size(0)):
            mask[i,0,x_index[i]:x_index[i]+args.mks,y_index[i]:y_index[i]+args.mks,z_index[i]:z_index[i]+args.mks] = 0.0
#            for k in range(inputs_str.size(2)):
#                if torch.all(inputs_str[i:i+1,0:1,k:k+1,:]) or torch.all(inputs_str[i:i+1,1:2,k:k+1,:]):

                
            
        inputs = inputs*mask
        inputs = inputs.to(device)
        y_true = y_true.to(device)
#        pdb.set_trace()
        inputs_str = inputs_str.to(device)
        y_true_str = y_true_str.to(device)
#        inputs_str.requires_grad = True


        # run inputs through the model to produce a warped image and flow field
        y_pred,y_pred_str = model(inputs,inputs_str)
        # calculate total loss
        loss = 0
        loss_list = []
        # with torch.autograd.set_detect_anomaly(True):


        for n, loss_function in enumerate(losses):
#            pdb.set_trace()
            y_true.requires_grad = True
            curr_loss = loss_function(y_true, y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

#        for i in range(len(args.organ_names)-1):
#        pdb.set_trace()
#        ssim = SSIM(data_range = 1.0,device = device)
#        str_loss = 0
#        for organ in range(5):
#            pr_str = y_pred_str
#            ssim.update((y_pred_str,y_true_str))
#            str_loss +=ssim.compute()
#            ssim.reset()
#        loss_list.append(str_loss/5)
#        loss +=str_loss/5

        curr_loss = str_loss(y_true_str,y_pred_str)*0.01
        loss_list.append(curr_loss.item())
        loss +=curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # get compute time
        epoch_step_time.append(time.time() - step_start_time)
#        pdb.set_trace()
    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

    # save model checkpoint
    tmp_loss = np.mean(epoch_loss,axis = 0)
    if tmp_loss[0] <=best_mae_loss or tmp_loss[2]<=best_ssim_loss:
        model.save(os.path.join(model_dir, f"{epoch:04d}_mae_{tmp_loss[0]:.4e}_ssim_{tmp_loss[2]:.4e}_.pt"))
        best_mae_loss = tmp_loss[0]
        best_ssim_loss = tmp_loss[2]

# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
