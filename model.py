import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch import nn, optim 
import numpy as np 
import matplotlib.pyplot as plt 
import utils

def tv(x, w=[1,1,0]):
    # w is the weight for different dimensions
    # by default, the weight of the third dimension is set to 0, implying only
    # spatial TV is considered and channel TV is ignored. 
    return  w[0]*(x[1:,:,:] - x[:-1,:,:]).abs().mean() + \
        w[1]*(x[:,1:,:] - x[:,:-1,:]).abs().mean() + \
        w[2]*(x[:,:,1:] - x[:,:,:-1]).abs().mean()

def n_mode_prod(ten, mat, n):
    ''' n-mode product
    ten is a tensor with shape [I1, I2, ..., In, ..., IN].
    max is a matrix with shape [J, In].
    '''
    import string
    letter_list = string.ascii_letters
    ndim = len(ten.shape)
    assert ndim<=51, 'The dimension of input tensor must be less than 51.'
    
    index4ten = letter_list[:ndim]
    index4mat = 'Z'+letter_list[n-1]
    index4out = index4ten
    index4out = index4out.replace(index4out[n-1],'Z')
    
    equ = index4ten+','+index4mat+'->'+index4out
    output = torch.einsum(equ, [ten, mat])
    
    return output

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

def get_act(act):
    if act == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01, inplace=False)
    elif act == 'relu':
        return  nn.ReLU()
    elif act == 'prelu':
        return  nn.PReLU()
    elif act == 'mish':
        return  nn.Mish()
    elif act == 'tanh':
        return  nn.Tanh()
    elif act == 'sigmoid':
        return  nn.Sigmoid()
    elif act == 'sin':
        return  Sine() 
    elif act == 'identity':
        return  nn.Identity()
        
        
class MultiNonLinear(nn.Module):
    def __init__(self, dim_in, dim_out, act=None):
        super().__init__()
        weight = []
        for temp_dim_in, temp_dim_out in zip(dim_in, dim_out):
            weight.append(torch.Tensor(temp_dim_in, temp_dim_out))
        self.weight = nn.ParameterList(weight)
        self.reset_parameters(negative_slope=0.01)
        
        if len(act)==1:
            act = len(dim_in)*act
        
        self.act = []
        for i in range(len(act)):
            self.act.append(get_act(act[i]))
        self.act = nn.ModuleList(self.act)
        
    def reset_parameters(self, negative_slope):
        for i in range(len(self.weight)):
            nn.init.kaiming_uniform_(self.weight[i], 
                                      a=negative_slope, 
                                      mode='fan_in', 
                                      nonlinearity='leaky_relu')
    
    def forward(self, x):
        for n in range(x.ndim):
            x = self.act[n](n_mode_prod(x, self.weight[n], n+1))
        return x 

class StackedTuckerDecomposition(nn.Module): 
    def __init__(self, shape, ratio, nlayer, act='mish'):
        super().__init__()
        # initialize core tensor
        min_shape = [int(shape[i]*ratio[i]) for i in range(len(shape))]
        if min_shape[-1]>=10:
            stdv = 1/min_shape[-1]**(1/2)
        else:
            stdv = 0.001
        self.core = nn.Parameter(torch.Tensor(*min_shape))
        nn.init.uniform_(self.core, -stdv, stdv)
        
        # initialize MultiNonLinear 
        layer_shape_mode = []
        for i in range(len(min_shape)):
            layer_shape_mode.append(torch.linspace(min_shape[i], shape[i], nlayer+1))
        layer_shape  = []
        for i in range(nlayer+1):
            layer_shape.append(
                [int(layer_shape_mode[j][i]) for j in range(len(layer_shape_mode))])

        # build network
        net = []
        for i in range(nlayer):
            if i<nlayer-1:
                temp_act = len(shape)*[act]
            elif i==nlayer-1:
                temp_act = (len(shape)-1)*[act]
                temp_act.append('identity')
            net.append(MultiNonLinear(layer_shape[i+1], layer_shape[i], temp_act))
        self.net = nn.Sequential(*net)
                       
    def forward(self):
        output = self.net(self.core)
        return output

class STDTV:
    def __init__(self, 
                 data,
                 max_iter,
                 lr,
                 alpha, # weight for 3DTV regularizer
                 beta, # parameter for 3DTV controlling weight for three directions
                 s, # rank ratio for the minimum rank
                 nlayer, # the number of layers for STD
                 act, # activation function
                 smooth_coef = 0.9,
                 clamp = True, # whether clamp the data into [0,1]. For most imagery data, set clamp to True.
                 print_stat = True,
                 print_img = True,
                 print_channel = [30,20,10]
                 ):
        
        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # data
        data_tensor = {'Input': torch.tensor(data['Input']).float().to(self.device),
                       'Mask': torch.tensor(data['Mask']).float().to(self.device) if data['Mask'] is not None else None
                       }
        self.data = data
        self.data_tensor = data_tensor
        self.data_shape = data['Input'].shape # [H,W,C]

        # STD model
        self.s = s
        self.nlayer = nlayer
        self.act = act
        self.model = StackedTuckerDecomposition(self.data_shape,# original data shape
                                                s, # rank ratio
                                                nlayer, # number of layers
                                                act # activation function
                                                ).to(self.device)
        # optimization
        self.max_iter = max_iter
        self.lr = lr
        self.smooth_coef = smooth_coef
        self.clamp = clamp
        self.optimizier = optim.Adam(self.model.parameters(), lr=self.lr) 
                
        # 3DTV
        self.alpha = alpha
        self.beta = beta
        
        # loss function
        self.loss_fn = nn.MSELoss()
        
        # print 
        self.print_stat = print_stat
        self.print_img = print_img
        self.print_channel = print_channel
        
    def train(self):
        # obtain data
        observation = self.data_tensor['Input']
        mask = self.data_tensor['Mask']
        
        # initialize output
        output_np = None
        
        # record metrics
        if self.data['GT'] is not None:
            psnr_list = []
        
        for iter_ in range(1,self.max_iter+1):
            # forward propagation (reconstruction data)
            output = self.model()
            # calculate loss
            loss = self.loss_fn(output*mask,observation*mask)
            if self.alpha != 0:
                loss += self.alpha*tv(output,self.beta)
            # backward propagation (compute gradient & update parameters)
            self.optimizier.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizier.step()
            
            # post-processing
            with torch.no_grad():
                # orthogonal projection
                output[mask == 1] = observation[mask == 1]
                # clamp data into [0,1]
                if self.clamp:
                    output.clamp_(0,1)
            # convert as array format
            if output_np is None:
                output_np = output.cpu().detach().numpy()
            else:
                output_np = self.smooth_coef*output_np + (1-self.smooth_coef)*output.cpu().detach().numpy()
            
            # record metrics
            if self.data['GT'] is not None:
                psnr_list.append(utils.eval_metrics(self.data['GT'], output_np, ['psnr3d'])[0])
                
            # print stat & img
            if iter_==1 or iter_ % 100 == 0 or iter==self.max_iter:
                if self.print_stat:
                    if self.data['GT'] is None:
                        print('iteration: %05d, loss=%.2e'%(iter_, loss.data))
                    else:
                        # psnr_val = utils.eval_metrics(self.data['GT'], output_np, ['psnr3d'])[0]
                        # print('iteration: %05d, loss=%.2e, PSNR=%.4f'%(iter_, loss.data, psnr_val))
                        print('iteration: %05d, loss=%.2e, PSNR=%.4f'%(iter_, loss.data, psnr_list[-1]))
                
                if self.print_img:
                    plt.subplot(121)
                    plt.imshow(np.clip(self.data['Input'][:,:,self.print_channel],0,1))
                    plt.title('Observed')
                    plt.subplot(122)
                    plt.imshow(np.clip(output_np[:,:,self.print_channel],0,1))
                    plt.title('Recovered')
                    plt.show()
                    
        self.output = output_np
        self.psnr_list = psnr_list
              