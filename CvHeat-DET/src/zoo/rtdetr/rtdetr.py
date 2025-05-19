"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', 'DETR', 'videoDETR']


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 

@register
class videoDETR(nn.Module):
    __inject__ = ['encoder', 'decoder']

    def __init__(self, encoder, decoder):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        x = self.encoder(x)
        x = x.permute(0, 2, 1, 3, 4)
        b, t, c, h, w =x.shape
        x = x.contiguous().view(b*t, c, h, w)
        output = self.decoder(x, targets)
        return output

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
    
@register
class DETR(nn.Module):
    __inject__ = ['encoder', 'decoder']

    def __init__(self, encoder, decoder):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
    

@register
class videoDETR2Dconv(nn.Module):
    __inject__ = ['encoder', 'decoder']

    def __init__(self, encoder, decoder):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        x = self.encoder(x)
        output = self.decoder(x, targets)
        return output

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 