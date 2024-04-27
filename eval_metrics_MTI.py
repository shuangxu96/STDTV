# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import utils
from glob import glob
from skimage.io import imread
from os.path import join
from scipy.io import loadmat, savemat



task = 'MTI_deadline' # or 'MTI_decloud'
method = ['STDTV']
gt_path = 'data/%s'%(task)
test_path = ['result/%s/%s'%(task,temp_method) for temp_method in method]
    
gt_f = glob(join(gt_path,'*.mat'))
metric_name = ['PSNR3D', 'SSIM3D', 'ERGAS', 'SAM']

num_samples = len(gt_f)
num_methods = len(method)
num_metrics = len(metric_name)


file_index = 0
file_name = gt_f[file_index]
x = utils.im2double(loadmat(file_name)['GT']) # gt
hh,ww,cc,tt = x.shape
print(file_name)

metrics = np.zeros((tt,num_metrics,num_methods))

for jj in range(num_methods):
    current_method = test_path[jj]
    file_name = gt_f[file_index].replace(gt_path,test_path[jj])
    y = utils.im2double(loadmat(file_name)['img_recon'])
    for time_node in range(tt):
        metrics[time_node,:,jj] = utils.eval_metrics(x[:,:,:,time_node], y[:,:,:,time_node], metric_name)

savemat('result/%s/metrics.mat'%(task), {'metrics':metrics})
np.savetxt('result/%s/metrics.txt'%(task), 
           np.reshape(metrics, [tt*num_metrics,num_methods]), 
           fmt='%.6f', delimiter=',')
for jj in range(num_methods):
    np.savetxt(join(test_path[jj],'metrics.txt'), metrics[:,:,jj], fmt='%.6f', delimiter=',')
    
