#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

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
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import argparse

# third party
import numpy as np
#import nibabel as nib
import torch

import itk
import time

from DicomfromArray import *

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



# parse commandline args
parser = argparse.ArgumentParser()
#parser.add_argument('--moving', required=True, help='moving image (source) filename')
#parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--img-list', default = './data_pre/test_lung_index_v5.txt', help='line-seperated list of training files')
parser.add_argument('--moved', default=True, help='warped image output filename')
parser.add_argument('--model',default = './models5_3/1500.pt',  help='pytorch model for nonlinear registration')
parser.add_argument('--warp',default = True,  help='output warp deformation filename')
parser.add_argument('-g', '--gpu',default = '0,2', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

args = parser.parse_args()

with open(args.img_list,'r') as file:
    content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]

bidir = args.bidir
# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel


# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)

model.to(device)
model.eval()

Mseloss = vxm.losses.MSE().loss
Maeloss = vxm.losses.MAE().loss
Nccloss =vxm.losses.NCC(27).loss
Diceloss = vxm.losses.Dice().loss

msetotal1 = []
maetotal = []
msetotal = []
timetotal = []
for i in range(len(filelist)):
    test_images, moving_max,fixed_max = vxm.py.utils.load_volfile(filelist[i], add_batch_axis=True, add_feat_axis=add_feat_axis)
    test_images = torch.Tensor(test_images)
    moving = test_images[0:1,:].squeeze(-1)

    moving = moving[None,:]
    fixed = test_images[1:2,:].squeeze(-1)
    fixed = fixed[None,:]
    movename = filelist[i].split(':')[0]
#fixed, fixed_affine = vxm.py.utils.load_volfile(
#    args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

    moving = moving.to(device)
    fixed = fixed.to(device)
    inputs = torch.cat((moving,fixed),axis = 1)

    start = time.time()
    moved, warp = model(inputs, registration=True)
    end = time.time()

    fixed = fixed*(fixed_max+1000)-1000
    moved = moved*(fixed_max+1000)-1000
    moving = moving*(moving_max+1000)-1000
    mse1 = Mseloss(fixed,moved)
    mae = Maeloss(fixed,moved)
    
 

    mse1 = mse1.detach().cpu().numpy()
    mae = mae.detach().cpu().numpy()
    fixed = fixed.squeeze().detach().cpu().numpy()
    moved = moved.squeeze().detach().cpu().numpy()
    moving = moving.squeeze().detach().cpu().numpy()

    maetotal.append(mae)
    #msetotal.append(mse)
    msetotal1.append(mse1)
    timetotal.append(end-start)

    print('mse1: '+str(mse1))

    print('mae: '+str(mae))
    print('time elapsed: '+str(end-start))

    if  args.moved:

        itkmoved = itk.image_view_from_array(moved)
        fname_moved = './test_results/1500_5_3/'+movename+'moved'
        itk.imwrite(itkmoved,fname_moved+'.mhd')

        itkmoving = itk.image_view_from_array(moving)
        fname_moving = './test_results/1500_5_3/'+movename+'moving'
        itk.imwrite(itkmoving,fname_moving+'.mhd')

        itkfixed = itk.image_view_from_array(fixed)
        fname_fixed = './test_results/1500_5_3/'+movename+'fixed'
        itk.imwrite(itkfixed,fname_fixed+'.mhd')


# save warp
    if  args.warp:
        warp = warp.permute(0,2,3,4,1)
        warp = warp.detach().cpu().numpy().squeeze()
        itkwarp = itk.image_view_from_array(warp)
        itk.imwrite(itkwarp,'./test_results/1500_5_3/'+movename+'warp.mhd')


