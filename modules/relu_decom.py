#!/usr/bin/env python

import pdb
import math

import numpy as np

import torch
from torch import nn

from .utils import build_montage, normalize
    
class ReLULayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with ReLU non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return nn.functional.relu(self.linear(input))
    
class PosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            #assert fn_samples is not None
            fn_samples = sidelength
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = 4

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)
    
class INR(nn.Module):
    def __init__(self, in_features,
                 hidden_features, hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode

        self.feat_per_channel = [2,2]

        self.coord_input_layer = nn.ModuleList(
            [nn.Linear(feat, hidden_features) for feat in self.feat_per_channel]
        )
        self.nonlin = ReLULayer

        self.coord_hidden_layers = 2
        self.coord_net = nn.ModuleList()
        for i in range(self.coord_hidden_layers):
            self.coord_net.append(self.nonlin(hidden_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))


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
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features,
                                     out_features , bias = True)
                        
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        #self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        #breakpoint()
        # if self.pos_encode:
        #     coords = self.positional_encoding(coords)
        hs = [self.forward_coord(coord, i) for i, coord in enumerate(coords)]
        h = self.forward_fusion(hs)
        
        sh = h.shape
                    
                    
        return h.reshape(-1,sh[-1])
    

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