#!/usr/bin/env python

import pdb
import math

import numpy as np

import torch
from torch import nn

from .utils import build_montage, normalize
    
class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    
class INR(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True , wire_tunable = False , real_gabor = False):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        if real_gabor:
            self.nonlin = RealGaborLayer
            dtype = torch.float32
        
        else:
            self.nonlin = ComplexGaborLayer
            hidden_features = int(hidden_features/np.sqrt(2))
            dtype = torch.cfloat
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        self.complex = True
        self.wavelet = 'gabor'    
        self.pos_encode = pos_encode

        self.feat_per_channel = [2,2]

        self.coord_input_layer = nn.ModuleList(
            [self.nonlin(feat, hidden_features, is_first=True ,omega0=first_omega_0,
                                  sigma0=scale) for feat in self.feat_per_channel]
        )
        #self.nonlin = ReLULayer

        self.coord_hidden_layers = 2
        self.coord_net = nn.ModuleList()
        for i in range(self.coord_hidden_layers):
            self.coord_net.append(self.nonlin(hidden_features, hidden_features, 
                                  is_first=False, omega0=first_omega_0,
                                  sigma0=scale))


        self.hidden_layers = 2
  
        self.fusion_operator = 'prod'
        
        # self.complex = False
            
        # if pos_encode:
        #     self.positional_encoding = PosEncoding(in_features=in_features,
        #                                            sidelength=sidelength,
        #                                            fn_samples=fn_samples,
        #                                            use_nyquist=use_nyquist)
        #     in_features = self.positional_encoding.out_dim
            
        self.net = nn.ModuleList()
        # self.net.append(self.nonlin(in_features, hidden_features, 
        #                           is_first=True, omega_0=first_omega_0,
        #                           scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega0=hidden_omega_0,
                                      sigma0=scale))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features,
                                     out_features , bias = True,dtype=dtype)
                        
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega0=hidden_omega_0,
                                      sigma0=scale))
        
        #self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        #breakpoint()
        # if self.pos_encode:
        #     coords = self.positional_encoding(coords)
        hs = [self.forward_coord(coord, i) for i, coord in enumerate(coords)]
        h = self.forward_fusion(hs)
        
        sh = h.shape
                    
                    
        return (h.reshape(-1,sh[-1])).real
    

    def forward_coord(self, coord, channel_id):
        h = self.coord_input_layer[channel_id](coord)

        for i in range(self.coord_hidden_layers):
            h = self.coord_net[i](h)
        
        return h
    
    def forward_fusion(self, hs):
        h = hs[0]
        for hi in hs[1:]:
            if self.fusion_operator == 'sum':
                h = h + hi
            elif self.fusion_operator == 'prod':
                h = h * hi
        #breakpoint()
        for i in range(self.hidden_layers+1):
            h = self.net[i](h)
        return h