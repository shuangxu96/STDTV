import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import utils
import numpy as np 

from glob import glob
from model import STDTV
from scipy.io import loadmat, savemat

# task 
task = 'HSI_inpainting'

# load data filenames
datapath = 'data/%s/'%(task)
datafile = glob(datapath+'*.mat')
datafile.sort()
datanum = len(datafile)

# savepath
savepath = 'result/%s/STDTV/'%(task)
os.makedirs(savepath, exist_ok=True)

# set metric
metric_name = ['PSNR3D', 'SSIM3D', 'ERGAS', 'SAM']
metrics = np.zeros((datanum, len(metric_name)))

# optimization parameters
max_iter = 10000
lr = 0.001

# 3DTV parameters
alpha = 1e-4 # weight for 3DTV regularizer
beta = [1,1,1]

# STD model parameters
s = [0.6,0.6,0.1] # rank ratio for the minimum rank

nlayer = 2 # the number of layers for STD
act = 'mish' # activation function

# post-processing parameters
smooth_coef = 0.9 #
clamp = True # whether clamp the data into [0,1]. For most imagery data, set clamp to True.

# debug printing
print_stat = True
print_img = True
print_channel = [30,20,10]

# carry out experiment
for i, current_data in enumerate(datafile):
    # load data
    mat = loadmat(current_data)
    GT = utils.im2double(mat['GT'])
    Input = utils.im2double(mat['Input'])
    Mask = mat['Mask']
    data = utils.set_data(Input, Mask, GT)
    # set model 
    model = STDTV(data,
                max_iter,
                lr,
                alpha, # weight for 3DTV regularizer
                beta, # parameter for 3DTV controlling weight for three directions
                s, # rank ratio for the minimum rank
                nlayer, # the number of layers for STD
                act, # activation function
                smooth_coef = smooth_coef,
                clamp = clamp, # whether clamp the data into [0,1]. For most imagery data, set clamp to True.
                print_stat = print_stat,
                print_img = print_img,
                print_channel = print_channel
                )
    # train
    model.train()
    metrics[i,:] = utils.eval_metrics(GT, model.output, metric_name)
    filename = current_data.split('\\')[-1]
    savemat(savepath+filename, {'img_recon':np.uint8(255*model.output)})
    print(metrics[i,:])                                                                             

np.savetxt(savepath+'metrics.txt', metrics, fmt='%.6f', delimiter=',')