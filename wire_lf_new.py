#!/usr/bin/env python

import os
import sys
from tqdm import tqdm
import importlib
import time
import glob

import argparse

import numpy as np
from scipy import io
from PIL import Image

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

from logger import PSNRLogger
def parse_argument():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=8 )
    parser.add_argument('--width', type=int, default=256 )
    
    parser.add_argument('--whole_epoch', type=int, default=100 )

    parser.add_argument('--data_dir', type=str , default="data/stanford_half/beans")
    parser.add_argument('--exp_dir', type=str , default="result/stanford_half/beans")
    
    parser.add_argument('--test_freq', type=int , default=10)
    parser.add_argument('--save_ckpt_path', type=int , default=100)
    parser.add_argument('--lr', type=float , default=5e-3) 
    parser.add_argument('--batch_size',type=int, default = 256*256,help='normalize input')
    
    parser.add_argument('--save_test_img', action='store_true')
    parser.add_argument('--wire_tunable', action='store_true')
    parser.add_argument('--real_gabor', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--test_img_save_freq', type=int , default=-1)
    
    parser.add_argument('--lr_preset', action='store_true')
    parser.add_argument('--lr_batch_preset', action='store_true')
    
    
    parser.add_argument('--nonlin', type=str , default="relu")
    parser.add_argument('--omega', type=int , default=5)
    parser.add_argument('--sigma', type=int , default=5)

    parser.add_argument("--gpu", default="0", type=str, help="Comma-separated list of GPU(s) to use.")

    opt = parser.parse_args()
    return opt 



def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    nonlin = opt.nonlin            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = opt.whole_epoch               # Number of SGD iterations
    
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3 
    
    tau = 3e1                   # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2               # Readout noise (dB)
    
    # Gabor filter constants.
    # We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
    omega0 = opt.omega           # Frequency of sinusoid
    sigma0 = opt.sigma           # Sigma of Gaussian
    
    # Network parameters
    hidden_layers = opt.depth      # Number of hidden layers in the MLP
    hidden_features = opt.width   # Number of hidden units per layer
         # Batch size
    
    # Read image and scale. A scale of 0.5 for parrot image ensures that it
    # fits in a 12GB GPU
#    im = utils.normalize(plt.imread('data/parrot.png').astype(np.float32), True)
#    im = cv2.resize(im, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
#    H, W, _ = im.shape
    
    # Create a noisy image
#    im_noisy = utils.measure(im, noise_snr, tau)


    if opt.lr_batch_preset:
        if opt.nonlin =="relu" or opt.nonlin =="relu_skip" or opt.nonlin =="relu_skip2":
            learning_rate =0.0005
            maxpoints = 8192
            
        elif opt.nonlin =="wire": 
            if opt.depth == 8:
                learning_rate =0.001
            else:
                learning_rate =0.005
            maxpoints = 65536
                
            
        elif opt.nonlin =="siren": 
            learning_rate =0.0005
            maxpoints = 8192
            
        elif opt.nonlin =="gauss": 
            learning_rate =0.005
            maxpoints = 65536
            
        elif opt.nonlin =="finer": 
            learning_rate =0.0005
            maxpoints = 65536
            
            
            
            
    else:
        learning_rate = opt.lr
        maxpoints = opt.batch_size       
    
    print(f"learning_rate : {learning_rate}")
    print(f"batch_size : {maxpoints}")
                
        
    # args
    norm_fac = 1
    st_norm_fac = 1
    rep = 1
    data_root = opt.data_dir
    save_dir  = opt.exp_dir
    
    logger_path = os.path.join(save_dir , 'log')
    test_path = os.path.join(save_dir , 'test')
    ckpt_path = os.path.join(save_dir, 'checkpoint')
    
    logger = PSNRLogger(logger_path , opt.exp_dir.split('/')[-1])
    logger.set_metadata("depth",opt.depth)
    logger.set_metadata("width",opt.width)
    dataset_name = opt.data_dir.split('/')[-1]
    logger.set_metadata("dataset_name",dataset_name)
    logger.set_metadata("model_info",nonlin)
    logger.set_metadata("lr",learning_rate)
    logger.set_metadata("batch_size",opt.batch_size)
    logger.set_metadata("omega",opt.omega)
    logger.set_metadata("omega",opt.sigma)
    
    
    
    logger.load_results()
    
    test_freq = opt.test_freq
    if opt.test_img_save_freq == -1:
        test_img_save_freq = test_freq
    else:
        test_img_save_freq = opt.test_img_save_freq
    paths =[]
    
    paths.append(logger_path)
    paths.append(test_path)
    paths.append(ckpt_path)
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)



    # load nelf data
    print(f"Start loading...")
    split='train'
    # center
    #breakpoint()
    uvst_whole = np.load(f"{data_root}/uvst{split}.npy") / norm_fac
    uvst_whole[:,2:] /= st_norm_fac

    # norm to 0 to 1
    uvst_min = uvst_whole.min()
    uvst_max = uvst_whole.max()
    uvst_whole = (uvst_whole - uvst_min) / (uvst_max - uvst_min) * 2 - 1.0

    image_path = os.path.join(data_root, 'images')

    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

    # 첫 번째 이미지 파일 읽기
    if image_files:
        first_image_path = os.path.join(image_path, image_files[0])
        with Image.open(first_image_path) as img:
            w, h = img.size
            img_w,img_h = img.size

        print(f"Width: {w}, Height: {h}")
    else:
        print("No image files found in the directory.")
        

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
        
    #breakpoint()
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
                    sidelength=sidelength,
                    wire_tunable=opt.wire_tunable,
                    real_gabor=opt.real_gabor)
    

    ckpt_paths = glob.glob(os.path.join(ckpt_path,"*.pth"))
    #breakpoint()
    load_epoch = 0
  
    if len(ckpt_paths) > 0:
        for path in ckpt_paths:
            print(ckpt_path)
            ckpt_id = int(os.path.basename(path).split("ep")[1].split(".")[0])
            load_epoch = max(load_epoch, ckpt_id)
        ckpt_name = f"./{path}/nelf-{load_epoch}.pth"
        # ckpt_name = f"{self.checkpoints}nelf-{self.fourier_epoch}.pth"
        print(f"Load weights from {ckpt_name}")

        ckpt = torch.load(ckpt_name)

        model.load_state_dict(ckpt)

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
    
    if (opt.nonlin == 'relu') or (opt.nonlin =='relu_skip'):
         scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.995) 

        
    uvst_whole = torch.tensor(uvst_whole).cuda() 
    color_whole = torch.tensor(color_whole).cuda()
    
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
    

    divergence_count = 0
   
    indices = torch.randperm(train_size)
    for i in tbar:
        epoch = load_epoch + i + 1
#        indices = torch.randperm(H*W)
        if opt.benchmark:
            loop_start = torch.cuda.Event(enable_timing=True)
            loop_end = torch.cuda.Event(enable_timing=True)

            # 반복문 시작 전에 기록
            loop_start.record()
        
        if not ((epoch %test_freq == 0) and opt.benchmark):    
            for b_idx in range(0, train_size, maxpoints):
                b_indices = indices[b_idx:min(train_size, b_idx+maxpoints)]
                b_coords = uvst_whole[b_indices, ...]
                b_indices = b_indices
                pixelvalues = model(b_coords)
                

                loss = ((pixelvalues - color_whole[b_indices, :])**2).mean() 
            
                optim.zero_grad()
                loss.backward()
                optim.step()
        else:
            avg_forward_time = 0
            avg_backward_time = 0
            whole_batch_iter = train_size//maxpoints
            #print(f"whole_batch_iter : {whole_batch_iter}")
            for b_idx in range(0, train_size, maxpoints):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                b_indices = indices[b_idx:min(train_size, b_idx+maxpoints)]
                b_coords = uvst_whole[b_indices, ...]
                b_indices = b_indices  # 실질적인 연산이 없는 줄, 타이밍에 포함
        
                start.record()
                pixelvalues = model(b_coords)
                end.record()
                torch.cuda.synchronize()
                
                avg_forward_time += (start.elapsed_time(end) / whole_batch_iter)

    
                loss = ((pixelvalues - color_whole[b_indices, :])**2).mean()
            
                start.record()
                optim.zero_grad()
                loss.backward()
                optim.step()
                end.record()
                torch.cuda.synchronize()
                avg_backward_time += (start.elapsed_time(end) / whole_batch_iter)
            
        if  ((epoch %test_freq == 0) and opt.benchmark):# 반복문 끝난 후 시간 기록
            loop_end.record()
            torch.cuda.synchronize()
            #logger.set_metadata("per_epoch_whole_time",loop_start.elapsed_time(loop_end))
            #print(f"avg_forward_time : {avg_forward_time}, avg_backward_time :  {avg_backward_time},whole_time : {loop_start.elapsed_time(loop_end)}")
            logger.push_time(avg_forward_time , avg_backward_time ,loop_start.elapsed_time(loop_end),epoch)
        
    #time_array[epoch] = time.time() - init_time

        with torch.no_grad():
    #            pixelvalues_val = model(uvst_whole_val)
    #            mse_loss_array[epoch] = ((color_whole_val - pixelvalues_val)**2).mean().item()

    #            mse_loss_array[epoch] = ((gt_noisy - rec)**2).mean().item()
    #            mse_array[epoch] = ((gt - rec)**2).mean().item()
    #            im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
    #            im_rec = rec.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
            if (epoch % opt.save_ckpt_path == 0) and (epoch != 0):
                cpt_path = ckpt_path + f"ep{epoch}.pth"
                torch.save(model.state_dict(), cpt_path)

            if epoch % test_freq ==0:
                avg_inference_time = 0
                avg_backward_time = 0
                #print(f"epoch : {epoch}")
                i = 0
                count = 0
                psnr_arr = []
                val_size = uvst_whole_val.shape[0] / (img_w*img_h)
                #breakpoint()
                while i < uvst_whole_val.shape[0]:
                    #print(i)
                    end = i+img_w*img_h
                    uvst = uvst_whole_val[i:end]
                    uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    pred_color = model(uvst)
                    end_time.record()
                    torch.cuda.synchronize()
                    avg_inference_time += (start_time.elapsed_time(end_time) / val_size)
                    gt_color   = color_whole_val[i:end]
                    
                    pred_img = pred_color.reshape((img_h,img_w,3)).permute((2,0,1))
                    gt_img   = torch.tensor(gt_color).reshape((img_h,img_w,3)).permute((2,0,1))

                    if epoch % test_img_save_freq == 0:
                        torchvision.utils.save_image(pred_img,f"{test_path}/test_{count}.png")
                        torchvision.utils.save_image(gt_img,f"{test_path}/gt_{count}.png")

                    pred_color = pred_color.cpu().numpy()
                    psnr = peak_signal_noise_ratio(gt_color, pred_color, data_range=1)
        #                   ssim = structural_similarity(gt_color.reshape((img_h,img_w,3)), pred_color.reshape((img_h,img_w,3)), data_range=pred_color.max() - pred_color.min(),multichannel=True)
        #                   lsp  = self.lpips(pred_img.cpu(),gt_img)
                    psnr_arr.append(psnr)
                    #print(psnr)
        #                   s.append(ssim)
        #                   l.append(np.asscalar(lsp.numpy()))
                    #breakpoint()
                    i = end
                    count+=1
                
                logger.push_infer_time(avg_inference_time ,epoch)    
                print(f"infer time : {avg_inference_time:.2f}")
                
                whole_psnr = 0
                for psnr in psnr_arr:
                    whole_psnr += psnr
                psnr_result = whole_psnr/count
                logger.push(psnr_result , epoch)
                
                psnr_arr_rounded = [f"{psnr:.2f}" for psnr in psnr_arr]

                print(f"epoch : {epoch:.2f} , PSNR -> avg : {psnr_result:.2f}  all : {psnr_arr_rounded}")
                

                
                logger.save_results()

                # for name, param in model.named_parameters():
                #     if 'omega_0' in name or 'scale_0' in name:
                #         print(f'Epoch {epoch}, {name}: {param.item()}')

                # cpt_path = os.path.join(ckpt_path,f"ep{epoch}.pth")
                # torch.save(model.state_dict(), cpt_path)
                
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
    

   

    # if posencode:
    #     nonlin = 'posenc'
        
    # mdict = {
    #         #'rec': best_img,
    #         # 'gt': im,
    #         # 'im_noisy': im_noisy,
    #         # 'mse_noisy_array': mse_loss_array.detach().cpu().numpy(), 
    #          'mse_array': mse_array.detach().cpu().numpy(),
    #          'time_array': time_array.detach().cpu().numpy()}
    
#    os.makedirs('results/denoising', exist_ok=True)
#    io.savemat('results/denoising/%s.mat'%nonlin, mdict)

#    print('Best PSNR: %.2f dB'%utils.psnr(im, best_img))

    




if __name__ =="__main__":
    opt = parse_argument()
    
    run(opt)