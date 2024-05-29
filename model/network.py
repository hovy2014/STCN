"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x

class Decoder_MBO(nn.Module):
    def __init__(self, valuedim=512, headdim=256):
        super().__init__()
        self.compress = ResBlock(valuedim*2, headdim)
        self.up_16_8 = UpsampleBlock(128, headdim, headdim) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(48, headdim, headdim) # 1/8 -> 1/4

        self.pred = nn.Conv2d(headdim, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x

class Decoder_MB2(nn.Module):
    def __init__(self, valuedim=512, headdim=256):
        super().__init__()
        self.compress = ResBlock(valuedim*2, headdim)
        self.up_16_8 = UpsampleBlock(32, headdim, headdim) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(24, headdim, headdim) # 1/8 -> 1/4

        self.pred = nn.Conv2d(headdim, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x

class MemoryReader(nn.Module):
    def __init__(self, no_aff_amp=True):
        super().__init__()
        self.no_aff_amp = no_aff_amp
 
    def get_affinity(self, mk, qk):
        with torch.cuda.amp.autocast(enabled=not self.no_aff_amp):
            if self.no_aff_amp:
                mk = mk.float()
                qk = qk.float()
            B, CK, T, H, W = mk.shape
            mk = mk.flatten(start_dim=2)
            qk = qk.flatten(start_dim=2)

            # See supplementary material
            a_sq = mk.pow(2).sum(1).unsqueeze(2)
            ab = mk.transpose(1, 2) @ qk

            affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
            
            # softmax operation; aligned the evaluation style
            maxes = torch.max(affinity, dim=1, keepdim=True)[0]
            x_exp = torch.exp(affinity - maxes)
            x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
            affinity = x_exp / x_exp_sum 
        
        return affinity if not self.no_aff_amp else affinity.half()

    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out


class STCN(nn.Module):
    def __init__(self, single_object, backbone='res50res18', keydim=64, valuedim=512, headdim=256, no_aff_amp=True):
        super().__init__()
        self.single_object = single_object
        
        if backbone == 'res50res18':
            f16_channels = 1024

            self.key_encoder = KeyEncoder()
            if single_object:
                self.value_encoder = ValueEncoderSO()
            else:
                self.value_encoder = ValueEncoder()
            self.decoder = Decoder()
        elif backbone == 's0s0':
            f16_channels = 256

            self.key_encoder = KeyEncoder_MBO()
            if single_object:
                self.value_encoder = ValueEncoderSO_MBO(valuedim=valuedim)
            else:
                self.value_encoder = ValueEncoder_MBO(valuedim=valuedim)
            self.decoder = Decoder_MBO(valuedim=valuedim, headdim=headdim)
        elif backbone == 'v2v2':
            f16_channels = 96

            self.key_encoder = KeyEncoder_MB2()
            if single_object:
                self.value_encoder = ValueEncoderSO_MB2(valuedim=valuedim)
            else:
                self.value_encoder = ValueEncoder_MB2(valuedim=valuedim)
            self.decoder = Decoder_MB2(valuedim=valuedim, headdim=headdim)
        else:
            raise NotImplementedError

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(f16_channels, keydim=keydim)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(f16_channels, valuedim, kernel_size=3, padding=1)

        self.memory = MemoryReader(no_aff_amp)

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits

    def encode_key(self, frame): 
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, kf16, mask, other_mask=None): 
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask)
        return f16.unsqueeze(2) # B*512*T*H*W

    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, selector=None): 
        # q - query, m - memory
        # qv16 is f16_thin above
        affinity = self.memory.get_affinity(mk16, qk16)
        
        if self.single_object:
            logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
            prob = torch.sigmoid(logits)
        else:
            logits = torch.cat([
                self.decoder(self.memory.readout(affinity, mv16[:,0], qv16), qf8, qf4),
                self.decoder(self.memory.readout(affinity, mv16[:,1], qv16), qf8, qf4),
            ], 1)

            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob)
        prob = F.softmax(logits, dim=1)[:, 1:]

        return logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError


