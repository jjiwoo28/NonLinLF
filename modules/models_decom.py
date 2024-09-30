#!/usr/bin/env python

# Somewhat hacky way of importing

import pdb
import math

import numpy as np

import torch
from torch import nn

from . import gauss
from . import mfn
from . import relu
from . import siren
from . import wire
from . import wire2d
from . import relu_skip
from . import relu_skip2
from . import relu_decom
from . import wire_decom
from . import finer

layer_dict = {'gauss': gauss.GaussLayer,
              'relu': relu.ReLULayer,
              'siren': siren.SineLayer,
              'wire': wire.ComplexGaborLayer,
              'finer' : finer.FinerLayer}

def get_decom_INR( in_features, 
                coord_hidden_features = 256,
                coord_hidden_layers= 1,
                after_hidden_features= 256, 
                after_hidden_layers = 2, 
                before_hidden_features= 256, 
                before_hidden_layers = 2, 
                out_features = 3, 
                outermost_linear=True, 
                first_omega_0=30,
                hidden_omega_0=30,
                scale=10,
                split_input_nonlin1="relu",
                split_input_nonlin2="relu",
                after_nonlin="relu",
                before_nonlin="relu",
                R = 1,
                feat_per_channel  = [2,2]
                ):
    '''
        Function to get a class instance for a given type of
        implicit neural representation
        
        Inputs:
            nonlin: One of 'gauss', 'mfn', 'posenc', 'siren',
                'wire', 'wire2d'
            in_features: Number of input features. 2 for image,
                3 for volume and so on.
            hidden_features: Number of features per hidden layer
            hidden_layers: Number of hidden layers
            out_features; Number of outputs features. 3 for color
                image, 1 for grayscale or volume and so on
            outermost_linear (True): If True, do not apply nonlin
                just before output
            first_omega0 (30): For siren and wire only: Omega
                for first layer
            hidden_omega0 (30): For siren and wire only: Omega
                for hidden layers
            scale (10): For wire and gauss only: Scale for
                Gaussian window
            pos_encode (False): If True apply positional encoding
            sidelength (512): if pos_encode is true, use this 
                for side length parameter   
            fn_samples (None): Redundant parameter
            use_nyquist (True): if True, use nyquist sampling for 
                positional encoding
        Output: An INR class instance
    '''
    # if nonlin == 'wire':
    #     inr_mod = model_dict[nonlin]
    #     model = inr_mod.INR(in_features,
    #                         hidden_features,
    #                         hidden_layers,
    #                         out_features,
    #                         outermost_linear,
    #                         first_omega_0,
    #                         hidden_omega_0,
    #                         scale,
    #                         pos_encode,
    #                         sidelength,
    #                         fn_samples,
    #                         use_nyquist,
    #                         wire_tunable,
    #                         real_gabor)
    
    
    split_input1=layer_dict[split_input_nonlin1]
    split_input2=layer_dict[split_input_nonlin2]
    after=layer_dict[after_nonlin]
    before=layer_dict[before_nonlin]
    
    if split_input_nonlin1 != 'wire':
        model = INR(in_features,
                    coord_hidden_features=coord_hidden_features,
                    coord_hidden_layers=coord_hidden_layers,
                    after_hidden_features=after_hidden_features,
                    after_hidden_layers=after_hidden_layers,
                    before_hidden_features=before_hidden_features,
                    before_hidden_layers=before_hidden_layers,
                    out_features = out_features,
                    outermost_linear =outermost_linear,
                    first_omega_0 = first_omega_0,
                    hidden_omega_0 = hidden_omega_0 ,
                    scale = scale,
                    split_input_nonlin1=split_input1,
                    split_input_nonlin2=split_input2,
                    after_nonlin=after,
                    before_nonlin=before,
                    R = R,
                    feat_per_channel  = feat_per_channel
                    )
    else:
        #breakpoint()
        model = WIRE_INR(in_features,
                    coord_hidden_features=coord_hidden_features,
                    coord_hidden_layers=coord_hidden_layers,
                    after_hidden_features=after_hidden_features,
                    after_hidden_layers=after_hidden_layers,
                    before_hidden_features=before_hidden_features,
                    before_hidden_layers=before_hidden_layers,
                    out_features = out_features,
                    outermost_linear =outermost_linear,
                    first_omega_0 = first_omega_0,
                    hidden_omega_0 = hidden_omega_0 ,
                    scale = scale,
                    split_input_nonlin1=split_input1,
                    split_input_nonlin2=split_input2,
                    after_nonlin=after,
                    before_nonlin=before,
                    R = R,
                    feat_per_channel  = feat_per_channel
                    )
        
            
    return model


class INR(nn.Module):
    def __init__(self, in_features,
                 coord_hidden_features,
                 coord_hidden_layers,
                 after_hidden_features, 
                 after_hidden_layers, 
                 before_hidden_features, 
                 before_hidden_layers, 
                 out_features, 
                 outermost_linear=True,
                 first_omega_0=30, 
                 hidden_omega_0=30., 
                 scale=10.0,
                 split_input_nonlin1=relu.ReLULayer,
                 split_input_nonlin2=relu.ReLULayer,
                 after_nonlin=relu.ReLULayer,
                 before_nonlin=relu.ReLULayer,
                 R = 1,
                 feat_per_channel  = [2,2]
                 ):
        super().__init__()
      

        self.feat_per_channel = feat_per_channel
        self.split_input_nonlin1=split_input_nonlin1
        self.split_input_nonlin2=split_input_nonlin2
        self.after_nonlin=after_nonlin
        self.before_nonlin=before_nonlin
        
        self.coord_hidden_features = coord_hidden_features
        self.coord_hidden_layers = coord_hidden_layers
        self.after_hidden_features = after_hidden_features
        self.after_hidden_layers = after_hidden_layers
        self.before_hidden_features = before_hidden_features
        self.before_hidden_layers = before_hidden_layers
        self.reduce_method = "linear"


        self.coord_input_layer = nn.ModuleList(
            [nn.Linear(feat, coord_hidden_features) for feat in self.feat_per_channel]
        )
        
        #breakpoint()
        self.coord_net = nn.ModuleList()
        for i in range(self.after_hidden_layers):
            if (i == (self.after_hidden_layers -1)):
                
                self.coord_net.append(self.after_nonlin(after_hidden_features, after_hidden_features*R, 
                                    is_first=True, omega_0=first_omega_0,
                                    scale=scale))
                
            else:
                self.coord_net.append(self.after_nonlin(after_hidden_features, after_hidden_features, 
                                    is_first=True, omega_0=first_omega_0,
                                    scale=scale))
                


        
  
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

        for i in range(before_hidden_layers):
            self.net.append(self.before_nonlin(before_hidden_features, before_hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
            
        self.rgb_net = nn.Linear(before_hidden_features,
                                        out_features , bias = True)
        
      
        #self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        #breakpoint()
        # if self.pos_encode:
        #     coords = self.positional_encoding(coords)
        #breakpoint()
        hs = [self.forward_coord(coord, i) for i, coord in enumerate(coords)]
        h  = self.forward_fusion(hs)
        
        
        sh = h.shape
                    
        #breakpoint()            
        return h.reshape(-1,sh[-1])
    

    def forward_coord(self, coord, channel_id):
        h = self.coord_input_layer[channel_id](coord)

        for i in range(self.after_hidden_layers):
            h = self.coord_net[i](h)
        
        return h
    
    def forward_fusion(self, hs):
        h = hs[0]
        for hi in hs[1:]:
            h = h * hi
        #breakpoint()
        h_sh = h.shape
        
        #last_dim = h_sh[-1]
        
        # if self.before_hidden_features == 3:
        #     pass
            # # 마지막 차원을 3의 배수로 자릅니다.
            # truncated_size = (last_dim // 3) * 3
            # h = h[..., :truncated_size]

            # # 선택된 방법에 따라 다른 동작을 수행합니다.
            # if self.reduce_method == 'mean':
            #     # 평균 풀링
            #     h = h.reshape(*h_sh[:-1], -1, 3).mean(-2)
            # elif self.reduce_method == 'max':
            #     # 최대 풀링
            #     h = h.reshape(*h_sh[:-1], -1, 3).max(-2)
            # elif self.reduce_method == 'linear':
            #     # 선형 변환
            #     linear_transform = torch.nn.Linear(last_dim, 3).to(h.device)
            #     h = linear_transform(h)
            # else:
            #     raise ValueError(f"Unknown reduce method: {self.reduce_method}")
        #실험 -> only coord mlp 가 끝나도 가시 R전략을 사용하려면 아래 주석을 없애야함.
        if h_sh[-1] > self.before_hidden_features:
            h = h.reshape(*h_sh[:-1], self.before_hidden_features, -1).sum(-1)

        for i in range(self.before_hidden_layers):
            h = self.net[i](h)
            
        h = self.rgb_net(h)    
        
        return h
    


class WIRE_INR(nn.Module):
    def __init__(self, in_features,
                 coord_hidden_features,
                 coord_hidden_layers,
                 after_hidden_features, 
                 after_hidden_layers, 
                 before_hidden_features, 
                 before_hidden_layers, 
                 out_features, 
                 outermost_linear=True,
                 first_omega_0=30, 
                 hidden_omega_0=30., 
                 scale=10.0,
                 split_input_nonlin1=relu.ReLULayer,
                 split_input_nonlin2=relu.ReLULayer,
                 after_nonlin=relu.ReLULayer,
                 before_nonlin=relu.ReLULayer,
                 R = 1,
                 feat_per_channel  = [2,2]
                 ):
        super().__init__()
      

        self.feat_per_channel = feat_per_channel
        self.split_input_nonlin1=split_input_nonlin1
        self.split_input_nonlin2=split_input_nonlin2
        self.after_nonlin=after_nonlin
        self.before_nonlin=before_nonlin
        
        self.coord_hidden_features = int(coord_hidden_features/np.sqrt(2))
        self.coord_hidden_layers = coord_hidden_layers
        self.after_hidden_features = int(after_hidden_features/np.sqrt(2))
        self.after_hidden_layers = after_hidden_layers
        self.before_hidden_features = int(before_hidden_features/np.sqrt(2))
        self.before_hidden_layers = before_hidden_layers
        self.reduce_method = "linear"

        #breakpoint()
        self.coord_input_layer = nn.ModuleList(
            [nn.Linear(feat, self.coord_hidden_features ) for feat in self.feat_per_channel]
        )
        
        #breakpoint()
        self.coord_net = nn.ModuleList()
        for i in range(self.after_hidden_layers):
            if (i == (self.after_hidden_layers -1)):
                
                self.coord_net.append(self.after_nonlin(self.after_hidden_features, self.after_hidden_features*R, 
                                    is_first=True, omega_0=first_omega_0,
                                    scale=scale))
                
            else:
                self.coord_net.append(self.after_nonlin(self.after_hidden_features, self.after_hidden_features, 
                                    is_first=True, omega_0=first_omega_0,
                                    scale=scale))
                


        
  
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

        for i in range(before_hidden_layers):
            self.net.append(self.before_nonlin(self.before_hidden_features, self.before_hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        if not before_hidden_layers ==0:
            if outermost_linear:
                final_linear = nn.Linear(self.before_hidden_features,
                                        out_features , bias = True,dtype= torch.cfloat)
                            
                self.net.append(final_linear)
            else:
                self.net.append(self.nonlin(self.before_hidden_features, out_features, 
                                        is_first=False, omega_0=hidden_omega_0,
                                        scale=scale))
        else:
            #breakpoint()
            final_linear = nn.Linear(self.after_hidden_features,
                                        out_features , bias = True,dtype= torch.cfloat)
                            
            self.net.append(final_linear)
            
        #self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        #breakpoint()
        # if self.pos_encode:
        #     coords = self.positional_encoding(coords)
        #breakpoint()
        hs = [self.forward_coord(coord, i) for i, coord in enumerate(coords)]
        h ,temp= self.forward_fusion(hs)
        
        
        sh = h.shape
                    
                    
        return h.reshape(-1,sh[-1]).real , temp
    

    def forward_coord(self, coord, channel_id):
        h = self.coord_input_layer[channel_id](coord)

        for i in range(self.coord_hidden_layers):
            h = self.coord_net[i](h)
        
        return h
    
    def forward_fusion(self, hs):
        h = hs[0]
        for hi in hs[1:]:
            h = h * hi
        
        temp = h
        #breakpoint()
        h_sh = h.shape
        #last_dim = h_sh[-1]
        
        # if self.before_hidden_features == 3:
        #     pass
            # # 마지막 차원을 3의 배수로 자릅니다.
            # truncated_size = (last_dim // 3) * 3
            # h = h[..., :truncated_size]

            # # 선택된 방법에 따라 다른 동작을 수행합니다.
            # if self.reduce_method == 'mean':
            #     # 평균 풀링
            #     h = h.reshape(*h_sh[:-1], -1, 3).mean(-2)
            # elif self.reduce_method == 'max':
            #     # 최대 풀링
            #     h = h.reshape(*h_sh[:-1], -1, 3).max(-2)
            # elif self.reduce_method == 'linear':
            #     # 선형 변환
            #     linear_transform = torch.nn.Linear(last_dim, 3).to(h.device)
            #     h = linear_transform(h)
            # else:
            #     raise ValueError(f"Unknown reduce method: {self.reduce_method}")
        #실험 -> only coord mlp 가 끝나도 가시 R전략을 사용하려면 아래 주석을 없애야함.
        if h_sh[-1] > self.before_hidden_features:
            h = h.reshape(*h_sh[:-1], self.before_hidden_features, -1).sum(-1)

        for i in range(self.before_hidden_layers+1):
            h = self.net[i](h)
        return h , temp

