# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import utils
from glob import glob
from skimage.io import imread
from os.path import join
from scipy.io import loadmat, savemat


task = 'HSI_inpainting'
method = [ 'STDTV']

gt_path = 'data/%s'%(task)
test_path = ['result/%s/%s'%(task,temp_method) for temp_method in method]
    

gt_f = glob(join(gt_path,'*.mat'))
metric_name = ['PSNR3D', 'SSIM3D', 'ERGAS', 'SAM']

num_samples = len(gt_f)
num_methods = len(method)
num_metrics = len(metric_name)

metrics = np.zeros((num_samples,num_methods,num_metrics))

for ii in range(num_samples):
    file_name = gt_f[ii]
    x = utils.im2double(loadmat(file_name)['GT']) # gt
    if task == 'MTI_deadline' or task == 'MTI_decloud':
        hh,ww,_,_ = x.shape
        x = np.reshape(x, [hh,ww,-1],'F')
    print(file_name)
    for jj in range(num_methods):
        current_method = test_path[jj]
        file_name = gt_f[ii].replace(gt_path,test_path[jj])
        y = utils.im2double(loadmat(file_name)['img_recon'])
        metrics[ii,jj,:] = utils.eval_metrics(x, y, metric_name)

savemat('result/%s/metrics.mat'%(task), {'metrics':metrics})
np.savetxt('result/%s/metrics.txt'%(task), 
           np.reshape(metrics, [num_samples*num_methods,num_metrics]), 
           fmt='%.6f', delimiter=',')
for jj in range(num_methods):
    np.savetxt(join(test_path[jj],'metrics.txt'), metrics[:,jj,:], fmt='%.6f', delimiter=',')
    
