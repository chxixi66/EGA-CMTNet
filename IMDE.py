from functools import partial
import torch
from einops import einsum
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from typing import Callable, Optional, Tuple, Union, Any
import numpy as np
import numbers
import pywt  
import pywt.data

##########################################################################
##  Low-frequency Fusion
##########################################################################
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class."""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, q, k, v, minus=True):
        B, N, C = q.shape

        # Generate Q, K, V.
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Factorized attention.
        use_efficient = minus
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum(k_softmax, v, "b h n k, b h n v -> b h k v")
        factor_att = einsum(q, k_softmax_T_dot_v, "b h n k, b h k v -> b h n v")
        

        # Merge and reshape.
        if use_efficient:
            x = factor_att   
        else:
            x = v - factor_att   #residual connection
        x = x.transpose(1, 2).reshape(B, N, C)


        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MHCABlock(nn.Module):
    """Multi-Head Convolutional self-Attention block."""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=3,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            shared_cpe=None,
            shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.fuse = nn.Linear(dim * 2, dim)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = MLP(in_features=dim, hidden_features=dim * mlp_ratio,out_features=dim)

        self.norm2 = norm_layer(dim)

    def forward(self, q, k, v, minus=True):
        """foward function"""
        b, c, h, w = q.size(0), q.size(1), q.size(2), q.size(3)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        x = q + self.factoratt_crpe(q, k, v, minus)
        cur = self.norm2(x)
        x = x + self.mlp(cur)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return x
##########################################################################
##  High-frequency Fusion
##########################################################################
class GATE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Conv_gated = nn.Conv2d(3*dim, 3*dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.Sequential(
            nn.Conv2d(3 * dim, 3*dim, 1, 1, 0, bias=True).cuda(),
            nn.SiLU(inplace=True)
        )
        self.chnnel = nn.Sequential(
            nn.BatchNorm2d(3 * dim).cuda(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_ori = x
        x = self.act(x)
        x1 = self.Conv_gated(x_ori)
        x = x * x1
        x = self.chnnel(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                              padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)  
        
        concat = torch.cat([max_pool, avg_pool], dim=1)
    
        attn_map = self.sigmoid(self.conv(concat))
        
        return attn_map

class SpatialAttFusion(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super().__init__()
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, F_H_vis, F_H_ir):
        
        diff = F_H_vis - F_H_ir  # [B,C,H,W]
        
        attn_map = self.spatial_att(diff)  # [B,1,H,W]
        
        F_H_fuse = F_H_ir + F_H_vis * attn_map
        
        return F_H_fuse
##########################################################################
##  Inter-Modality Detail Enhancer (IMDE)
##########################################################################
class IMDE(nn.Module):
    def __init__(self, dim=32, num_heads=8, mlp_ratio=3, qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.dim = dim

        self.CFEM1 = MHCABlock(
            dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_scale=None,
            shared_cpe=None,
            shared_crpe=None,
        )
        self.CFEM2 = MHCABlock(
            dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_scale=None,
            shared_cpe=None,
            shared_crpe=None,
        )
        self.DFIM1 = MHCABlock(
            dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_scale=None,
            shared_cpe=None,
            shared_crpe=None,
        )
        self.DFIM2 = MHCABlock(
            dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_scale=None,
            shared_cpe=None,
            shared_crpe=None,
        )
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        self.gate1 = GATE(dim)
        self.gate2 = GATE(dim)
        self.SpatialAttFusion = SpatialAttFusion(in_channels=3*dim, kernel_size=7)

    def wavelet_transform(self, x, filters):
          b, c, h, w = x.shape
          self.origin_h, self.origin_w = h, w
          pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
          filters = filters.to(dtype=x.dtype, device=x.device)
          x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
          x = x.reshape(b, c, 4, h // 2, w // 2)

          return x
        
    def inverse_wavelet_transform(self, x, filters):
          b, c, _, h_half, w_half = x.shape
          pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
          filters = filters.to(dtype=x.dtype, device=x.device)
          x = x.reshape(b, c * 4, h_half, w_half)

          out_h, out_w = self.origin_h, self.origin_w
          output_padding_h = out_h - (h_half * 2 - (filters.shape[2] - 1 - 0))
          output_padding_w = out_w - (w_half * 2 - (filters.shape[3] - 1 - 0))
          output_padding_h = min(max(output_padding_h, 0), 1)
          output_padding_w = min(max(output_padding_w, 0), 1)


          x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=0, output_padding=(output_padding_h, output_padding_w))
          x = x[:, :, :out_h, :out_w]
          return x
        
    def create_wavelet_filter(self, wave, in_size, out_size, dtype=None):
          w = pywt.Wavelet(wave)
          dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
          dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
          dec_filters = torch.stack([
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # LL
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # LH
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # HL
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # HH
          ], dim=0)
          dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

          rec_hi = torch.tensor(w.rec_hi[::-1], dtype=torch.float32).flip(dims=[0])
          rec_lo = torch.tensor(w.rec_lo[::-1], dtype=torch.float32).flip(dims=[0])
          rec_filters = torch.stack([
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
          ], dim=0)
          rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

          return dec_filters, rec_filters
    def forward(self, vi_d, ir_d):
        vi = vi_d
        ir = ir_d
        self.origin_h, self.origin_w = vi.shape[2], vi.shape[3]
        dec_filters, rec_filters = self.create_wavelet_filter(wave='haar', in_size=self.dim, out_size=self.dim, dtype=vi.dtype)
        vi_wave = self.wavelet_transform(vi, dec_filters)
        ir_wave = self.wavelet_transform(ir, dec_filters)

        vi_A, vi_H, vi_V, vi_D = vi_wave[:, :, 0], vi_wave[:, :, 1], vi_wave[:, :, 2], vi_wave[:, :, 3]
        ir_A, ir_H, ir_V, ir_D = ir_wave[:, :, 0], ir_wave[:, :, 1], ir_wave[:, :, 2], ir_wave[:, :, 3]
        
        vi_A = self.conv1(vi_A)
        ir_A = self.conv2(ir_A)
        
        F_c1 = self.CFEM1(vi_A, ir_A, ir_A, minus=False)  
        F_c2 = self.CFEM2(ir_A, vi_A, vi_A, minus=False)    
        F_c = F_c1 + F_c2  #common

        F_d1 = self.DFIM1(F_c, vi_A, vi_A, minus=True) #difference
        F_d2 = self.DFIM2(F_c, ir_A, ir_A, minus=True)

        F_A = F_d1 + F_d2 + F_c  #low-frequency fusion

        vi_cat = torch.cat([vi_H, vi_V, vi_D], dim=1)
        ir_cat = torch.cat([ir_H, ir_V, ir_D], dim=1)

        vi_high = self.gate1(vi_cat)
        ir_high = self.gate2(ir_cat)    

        F_H = self.SpatialAttFusion(vi_high, ir_high)  #High-frequency fusion

        LL = F_A
        b, c, h_half, w_half = LL.shape
        LH = F_H[:, 0*c:1*c, :, :]  
        HL = F_H[:, 1*c:2*c, :, :]  
        HH = F_H[:, 2*c:3*c, :, :]  
        F_wave = torch.stack([LL, LH, HL, HH], dim=2)

        Et = self.inverse_wavelet_transform(F_wave, rec_filters)
        
        return Et

if __name__ == '__main__':
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    height = 643
    width = 481       
    
    IMDE = IMDE().to(device)

    batch_size = 1
    channels = 32
   

    structure_I = torch.randn(batch_size, channels, height, width).to(device)
    structure_V = torch.randn(batch_size, channels, height, width).to(device)

    mask = torch.randn(batch_size, 1, height, width).to(device) 
    mask_V = torch.randn(batch_size, 1, height, width).to(device) 
    mask_I = torch.randn(batch_size, 1, height, width).to(device)
   
    
    print(f"input : {structure_I.shape}")
    print(f"input : {structure_V.shape}")
    
    
    IMDE.eval()
    with torch.no_grad():
            fused_structure = IMDE(structure_I, structure_V)

            print(f"fusion output: {fused_structure.shape}")