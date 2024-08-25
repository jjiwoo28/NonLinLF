#!/usr/bin/env python

import pdb
import math

import numpy as np

import torch
from torch import nn

from .utils import build_montage, normalize
 
class INR(nn.Module):
    def __init__(self, in_features,
                 hidden_features, hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True,nonlin=None):
        super().__init__()
        self.pos_encode = pos_encode

        self.feat_per_channel = [2,2]

        self.coord_input_layer = nn.ModuleList(
            [nn.Linear(feat, hidden_features) for feat in self.feat_per_channel]
        )
        self.nonlin = nonlin

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