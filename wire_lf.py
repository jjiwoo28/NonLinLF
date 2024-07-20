#!/usr/bin/env python

import os
import sys
from tqdm import tqdm
import importlib
import time

import numpy as np
from scipy import io

import matplotlib.pyplot as plt
#plt.gray()

import cv2
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio

import torch
import torch.nn
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from pytorch_msssim import ssim

from modules import models
from modules import utils

if __name__ == '__main__':
    nonlin = 'wire'            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = 400               # Number of SGD iterations
    learning_rate = 5e-3       # Learning rate. 
    
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3 
    
    tau = 3e1                   # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2               # Readout noise (dB)
    
    # Gabor filter constants.
    # We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
    omega0 = 5.0           # Frequency of sinusoid
    sigma0 = 5.0           # Sigma of Gaussian
    
    # Network parameters
    hidden_layers = 4       # Number of hidden layers in the MLP
    hidden_features = 256   # Number of hidden units per layer
    maxpoints = 256*256     # Batch size
    
    # Read image and scale. A scale of 0.5 for parrot image ensures that it
    # fits in a 12GB GPU
#    im = utils.normalize(plt.imread('data/parrot.png').astype(np.float32), True)
#    im = cv2.resize(im, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
#    H, W, _ = im.shape
    
    # Create a noisy image
#    im_noisy = utils.measure(im, noise_snr, tau)


    # args
    norm_fac = 1
    st_norm_fac = 1
    rep = 1
    data_root = "data/stanford_half/beans"
    save_dir  = "result/stanford_half/beans"
    test_freq = 20

    # load nelf data
    print(f"Start loading...")
    split='train'
    # center
    uvst_whole = np.load(f"{data_root}/uvst{split}.npy") / norm_fac
    uvst_whole[:,2:] /= st_norm_fac

    # norm to 0 to 1
    uvst_min = uvst_whole.min()
    uvst_max = uvst_whole.max()
    uvst_whole = (uvst_whole - uvst_min) / (uvst_max - uvst_min) * 2 - 1.0

    # center color
    color_whole  = np.load(f"{data_root}/rgb{split}.npy")
    trans        = np.load(f"{data_root}/trans{split}.npy")
    intrinsic    = np.load(f"{data_root}/k{split}.npy")
    fdepth       = np.load(f"{data_root}/fdepth{split}.npy") # center object
    render_pose  = np.load(f"{data_root}/Render_pose{split}.npy")#render path spiral
    st_depth     = -fdepth

    uvst_whole  = np.concatenate([uvst_whole]*rep, axis=0)
    color_whole = np.concatenate([color_whole]*rep, axis=0)

    split='val'
    uvst_whole_val  = np.load(f"{data_root}/uvst{split}.npy") / norm_fac
    uvst_whole_val[:,2:] /= st_norm_fac
    color_whole_val = np.load(f"{data_root}/rgb{split}.npy")
    uvst_whole_val  = (uvst_whole_val - uvst_min) / (uvst_max - uvst_min) * 2 - 1.0

    trans_val        = np.load(f"{data_root}/trans{split}.npy")
    intrinsic_val    = np.load(f"{data_root}/k{split}.npy")
    fdepth_val       = np.load(f"{data_root}/fdepth{split}.npy") # center object
    render_pose_val  = np.load(f"{data_root}/Render_pose{split}.npy")#render path spiral
    st_depth_val     = -fdepth
    print("Stop loading...")

    uvst_whole = torch.tensor(uvst_whole) 
    color_whole = torch.tensor(color_whole)
#    uvst_whole_val = torch.tensor(uvst_whole_val) 
#    color_whole_val = torch.tensor(color_whole_val)
  
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
        
        if tau < 100:
#            sidelength = int(max(H, W)/3)
            sidelength = 128
        else:
#            sidelength = int(max(H, W))
            sidelength = 512
            
    else:
        posencode = False
#        sidelength = H
        sidelength = 512
        
    model = models.get_INR(
                    nonlin=nonlin,
                    in_features=4,
                    out_features=3, 
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    first_omega_0=omega0,
                    hidden_omega_0=omega0,
                    scale=sigma0,
                    pos_encode=posencode,
                    sidelength=sidelength)
        
    # Send model to CUDA
    model.cuda()
    
    print('Number of parameters: ', utils.count_parameters(model))
#    print('Input PSNR: %.2f dB'%utils.psnr(im, im_noisy))
    
    # Create an optimizer
#    optim = torch.optim.Adam(lr=learning_rate*min(1, maxpoints/(H*W)),
#                             params=model.parameters())
    optim = torch.optim.Adam(lr=learning_rate,params=model.parameters())
   
    # Schedule to reduce lr to 0.1 times the initial rate in final epoch
    scheduler = LambdaLR(optim, lambda x: 0.1**min(x/niters, 1))
    
 #   x = torch.linspace(-1, 1, W)
 #   y = torch.linspace(-1, 1, H)
    
 #   X, Y = torch.meshgrid(x, y, indexing='xy')
 #   coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
    
 #   gt = torch.tensor(im).cuda().reshape(H*W, 3)[None, ...]
 #   gt_noisy = torch.tensor(im_noisy).cuda().reshape(H*W, 3)[None, ...]
    
    mse_array = torch.zeros(niters, device='cuda')
    mse_loss_array = torch.zeros(niters, device='cuda')
    time_array = torch.zeros_like(mse_array)
    
 #   best_mse = torch.tensor(float('inf'))
 #   best_img = None
    
 #   rec = torch.zeros_like(gt)
    
    tbar = tqdm(range(niters))
    init_time = time.time()
    train_size = uvst_whole.shape[0]

    for epoch in tbar:
#        indices = torch.randperm(H*W)
        indices = torch.randperm(train_size)
       
        for b_idx in range(0, train_size, maxpoints):
#            b_indices = indices[b_idx:min(train_size, b_idx+maxpoints)]
#            b_coords = coords[:, b_indices, ...].cuda()
#            b_indices = b_indices.cuda()
#            pixelvalues = model(b_coords)

            b_indices = indices[b_idx:min(train_size, b_idx+maxpoints)]
            b_coords = uvst_whole[b_indices, ...].cuda()
            b_indices = b_indices
            pixelvalues = model(b_coords)
            
#            with torch.no_grad():
#                rec[:, b_indices, :] = pixelvalues
    
#            loss = ((pixelvalues - gt_noisy[:, b_indices, :])**2).mean() 
            loss = ((pixelvalues - color_whole[b_indices, :].cuda())**2).mean() 
           
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        time_array[epoch] = time.time() - init_time

        with torch.no_grad():
#            pixelvalues_val = model(uvst_whole_val)
#            mse_loss_array[epoch] = ((color_whole_val - pixelvalues_val)**2).mean().item()

#            mse_loss_array[epoch] = ((gt_noisy - rec)**2).mean().item()
#            mse_array[epoch] = ((gt - rec)**2).mean().item()
#            im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
#            im_rec = rec.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
            if epoch % test_freq ==0:
                i = 0
                count = 0
                while i < uvst_whole_val.shape[0]:
                    end = i+img_w*img_h
                    uvst = uvst_whole_val[i:end]
                    uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()
                    
                    pred_color = model(uvst)
                    gt_color   = color_whole_val[i:end]
                    
                    pred_img = pred_color.reshape((img_h,img_w,3)).permute((2,0,1))
                    gt_img   = torch.tensor(gt_color).reshape((img_h,img_w,3)).permute((2,0,1))

                    torchvision.utils.save_image(pred_img,f"{save_dir}/test_{count}.png")
                    torchvision.utils.save_image(gt_img,f"{save_dir}/gt_{count}.png")

                    pred_color = pred_color.cpu().numpy()
                    psnr = peak_signal_noise_ratio(gt_color, pred_color, data_range=1)
#                   ssim = structural_similarity(gt_color.reshape((img_h,img_w,3)), pred_color.reshape((img_h,img_w,3)), data_range=pred_color.max() - pred_color.min(),multichannel=True)
#                   lsp  = self.lpips(pred_img.cpu(),gt_img)

                    print(psnr)
#                   s.append(ssim)
#                   l.append(np.asscalar(lsp.numpy()))

                    i = end
                    count+=1

#                psnrval = -10*torch.log10(mse_loss_array[epoch])
#                tbar.set_description('%.1f'%psnrval)
#                tbar.refresh()
        
        scheduler.step()
        
#        imrec = rec[0, ...].reshape(H, W, 3).detach().cpu().numpy()
            
        #cv2.imshow('Reconstruction', imrec[..., ::-1])            
        #cv2.waitKey(1)
    
#        if (mse_array[epoch] < best_mse) or (epoch == 0):
#            best_mse = mse_array[epoch]
#            best_img = imrec
    
    if posencode:
        nonlin = 'posenc'
        
    mdict = {
            #'rec': best_img,
            # 'gt': im,
            # 'im_noisy': im_noisy,
            # 'mse_noisy_array': mse_loss_array.detach().cpu().numpy(), 
             'mse_array': mse_array.detach().cpu().numpy(),
             'time_array': time_array.detach().cpu().numpy()}
    
#    os.makedirs('results/denoising', exist_ok=True)
#    io.savemat('results/denoising/%s.mat'%nonlin, mdict)

#    print('Best PSNR: %.2f dB'%utils.psnr(im, best_img))
