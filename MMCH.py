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

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm.modules.mamba_simple import Mamba
import selective_scan_cuda
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

##########################################################################
## Simple Attention Module(SimAM)
##########################################################################
class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class Dila_Conv(nn.Module):
    def __init__(self, dim):
        super(Dila_Conv, self).__init__()

        out_1 = dim
        inp = dim*2

        self.conv_1 = nn.Conv2d(inp*2, out_1, padding=1, kernel_size=3, groups=out_1,
                                dilation=1)
        self.conv_2 = nn.Conv2d(inp*2, out_1, padding=2, kernel_size=3, groups=out_1,
                                dilation=2)
        self.conv_3 = nn.Conv2d(inp*2, out_1, padding=3, kernel_size=3, groups=out_1,
                                dilation=3)
        self.conv_4 = nn.Conv2d(inp*2, out_1, padding=4, kernel_size=3, groups=out_1,
                                dilation=4)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_1 * 4, out_1*2, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_1*2),
            nn.SiLU(inplace=True)
        )

    def forward(self, inp1, inp2):
        x = torch.cat([inp1, inp2], dim=1)
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)
        c4 = self.conv_4(x)
        cat = torch.cat([c1, c2, c3, c4], dim=1)
        fuse = self.fuse(cat)

        return fuse
##########################################################################
##  Efficient 2D Scanning.
##########################################################################
class EfficientMerge(torch.autograd.Function):  # [B, 4, C, H/w * W/w] -> [B, C, H*W]
    @staticmethod
    def forward(ctx, ys: torch.Tensor, ori_h: int, ori_w: int, step_size=2):
        B, K, C, L = ys.shape
        H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)
        ctx.shape = (H, W)
        ctx.ori_h = ori_h
        ctx.ori_w = ori_w
        ctx.step_size = step_size

        new_h = H * step_size
        new_w = W * step_size

        y = ys.new_empty((B, C, new_h, new_w))

        y[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, H, W)
        y[:, :, 1::step_size, ::step_size] = ys[:, 1].reshape(B, C, W, H).transpose(dim0=2, dim1=3)
        y[:, :, ::step_size, 1::step_size] = ys[:, 2].reshape(B, C, H, W)
        y[:, :, 1::step_size, 1::step_size] = ys[:, 3].reshape(B, C, W, H).transpose(dim0=2, dim1=3)

        if ori_h != new_h or ori_w != new_w:
            y = y[:, :, :ori_h, :ori_w].contiguous()

        y = y.view(B, C, -1)
        return y

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor):  # [B, C, H*W] -> [B, 4, C, H/w * W/w]

        H, W = ctx.shape
        B, C, L = grad_x.shape
        step_size = ctx.step_size

        grad_x = grad_x.view(B, C, ctx.ori_h, ctx.ori_w)

        if ctx.ori_w % step_size != 0:
            pad_w = step_size - ctx.ori_w % step_size
            grad_x = F.pad(grad_x, (0, pad_w, 0, 0))
        W = grad_x.shape[3]

        if ctx.ori_h % step_size != 0:
            pad_h = step_size - ctx.ori_h % step_size
            grad_x = F.pad(grad_x, (0, 0, 0, pad_h))
        H = grad_x.shape[2]
        B, C, H, W = grad_x.shape
        H = H // step_size
        W = W // step_size
        grad_xs = grad_x.new_empty((B, 4, C, H * W))

        grad_xs[:, 0] = grad_x[:, :, ::step_size, ::step_size].reshape(B, C, -1)
        grad_xs[:, 1] = grad_x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].reshape(B, C, -1)
        grad_xs[:, 2] = grad_x[:, :, ::step_size, 1::step_size].reshape(B, C, -1)
        grad_xs[:, 3] = grad_x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].reshape(B, C, -1)

        return grad_xs, None, None, None

class SelectiveScan(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True


        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

class EfficientScan(torch.autograd.Function):
    # [B, C, H, W] -> [B, 4, C, H * W] (original)
    # [B, C, H, W] -> [B, 4, C, H/w * W/w]
    @staticmethod
    def forward(ctx, x: torch.Tensor, step_size=2):  # [B, C, H, W] -> [B, 4, H/w * W/w]
        B, C, org_h, org_w = x.shape
        ctx.shape = (B, C, org_h, org_w)
        ctx.step_size = step_size

        if org_w % step_size != 0:
            pad_w = step_size - org_w % step_size
            x = F.pad(x, (0, pad_w, 0, 0))
        W = x.shape[3]

        if org_h % step_size != 0:
            pad_h = step_size - org_h % step_size
            x = F.pad(x, (0, 0, 0, pad_h))
        H = x.shape[2]

        H = H // step_size
        W = W // step_size

        xs = x.new_empty((B, 4, C, H * W))

        xs[:, 0] = x[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
        xs[:, 1] = x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 2] = x[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 3] = x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)

        xs = xs.view(B, 4, C, -1)
        return xs

    @staticmethod
    def backward(ctx, grad_xs: torch.Tensor):  # [B, 4, H/w * W/w] -> [B, C, H, W]

        B, C, org_h, org_w = ctx.shape
        step_size = ctx.step_size

        newH, newW = math.ceil(org_h / step_size), math.ceil(org_w / step_size)
        grad_x = grad_xs.new_empty((B, C, newH * step_size, newW * step_size))

        grad_xs = grad_xs.view(B, 4, C, newH, newW)

        grad_x[:, :, ::step_size, ::step_size] = grad_xs[:, 0].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, ::step_size] = grad_xs[:, 1].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)
        grad_x[:, :, ::step_size, 1::step_size] = grad_xs[:, 2].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, 1::step_size] = grad_xs[:, 3].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)

        if org_h != grad_x.shape[-2] or org_w != grad_x.shape[-1]:
            grad_x = grad_x[:, :, :org_h, :org_w]

        return grad_x, None

def cross_selective_scan_cross(
        x1: torch.Tensor = None,
        x2: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        step_size=2,
        G_conv: nn.Module = True,
):
    
    B, D, H, W = x1.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    # H * W
    ori_h, ori_w = H, W

    if G_conv is not None:
       x = G_conv(x1, x2)
    else:   
       x =x1*x2 +x1 + x2
    

    xs = EfficientScan.apply(x, step_size)  # [B, C, H*W] -> [B, 4, C, H//w * W//w]

    # H//w * W//w
    H = math.ceil(H / step_size)
    W = math.ceil(W / step_size)

    L = H * W

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)

    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, L)

    ori_h, ori_w = int(ori_h), int(ori_w)
    y = EfficientMerge.apply(ys, ori_h, ori_w, step_size)  # [B, 4, C, H//w * W//w] -> [B, C, H*W]

    H = ori_h
    W = ori_w
    L = H * W

    y = y.transpose(dim0=1, dim1=2).contiguous()
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

class SS2D_cross(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            # ======================
            forward_type="v2",
            # ======================
            step_size=2,
            Gconv=True,
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__() 
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv

        self.step_size = step_size
        self.d_model = d_model
        self.Gconv = Gconv

        if self.Gconv:
            self.dila_conv = Dila_Conv(dim=d_model)
        else:
            self.dila_conv = None

        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type =======================================
        self.forward_core = dict(
            v0=self.forward_corev0,
            v0_seq=self.forward_corev0_seq,
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            share_ssm=self.forward_corev0_share_ssm,
            share_a=self.forward_corev0_share_a,
        ).get(forward_type, self.forward_corev2)
        self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1

        # in proj =======================================
        self.in_proj1 = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act1: nn.Module = act_layer()
        self.act2: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version
    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        # def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        #     return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        self.selective_scan = selective_scan_fn

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    def forward_corev0_seq(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float()  # (b, k, d, l)
        dts = dts.contiguous().float()  # (b, k, d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1)  # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1)  # (k, d)

        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    def forward_corev0_share_ssm(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev0_share_a(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev2(self, x1: torch.Tensor, x2: torch.Tensor, nrows=-1, channel_first=False, step_size=2):
        nrows = 1
        if not channel_first:
            x1 = x1.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x1 = self.in_rank(x1)
            x2 = self.in_rank(x2)
        

        x = cross_selective_scan_cross(
            x1, x2, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, step_size=step_size, G_conv=self.dila_conv,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        xz1 = self.in_proj1(x1)
        xz2 = self.in_proj2(x2)
        if self.d_conv > 1:
            x1, z1 = xz1.chunk(2, dim=-1)  # (b, h, w, d)
            x2, z2 = xz2.chunk(2, dim=-1)  # (b, h, w, d)
            if not self.disable_z_act:
                z1 = self.act1(z1)
                z2 = self.act2(z1)
            x1 = x1.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
            x1 = self.act1(self.conv2d(x1))  # (b, d, h, w)
            x2 = self.act2(self.conv2d(x2))
        else:
            if self.disable_z_act:
                x1, z1 = xz1.chunk(2, dim=-1)  # (b, h, w, d)
                x2, z2 = xz2.chunk(2, dim=-1)
                x1 = self.act1(x1)
                x2 = self.act2(x2)
            else:
                xz1 = self.act1(xz1)
                xz2 = self.act2(xz2)
                x1, z1 = xz1.chunk(2, dim=-1)  # (b, h, w, d)
                x2, z2 = xz2.chunk(2, dim=-1)
        
        y = self.forward_core(x1, x2, channel_first=(self.d_conv > 1), step_size=self.step_size)
        y1 = y * z1
        y2 = y * z2
        y = y1 + y2 

        out = self.dropout(self.out_proj(y))
        return out

class ES2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        
        # Store original dtype
        original_dtype = x.dtype

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        
        # Convert back to original dtype before layer norm
        y = y.to(original_dtype)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
##########################################################################
##  Feature Enhancement
##########################################################################
class Enhancement_texture_LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        # conv.weight.size() = [out_channels, in_channels, kernel_size, kernel_size]
        super(Enhancement_texture_LDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)  # [12,3,3,3]

        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]).cuda()
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)  # [12,3,3,3]
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)  # [12,3]
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # [1]
        # print(self.learnable_mask[:, :, None, None].shape)

    def forward(self, x):
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        return out_diff

class Differential_enhance(nn.Module):
    def __init__(self, nf=48):
        super(Differential_enhance, self).__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()
        self.lastconv = nn.Conv2d(nf,nf//2,1,1)

    def forward(self, fuse, x1, x2):
        b,c,h,w = x1.shape
        sub_1_2 = x1 - x2
        sub_w_1_2 = self.global_avgpool(sub_1_2)
        w_1_2 = self.act(sub_w_1_2)
        sub_2_1 = x2 - x1
        sub_w_2_1 = self.global_avgpool(sub_2_1)
        w_2_1 = self.act(sub_w_2_1)
        D_F1 = torch.multiply(w_1_2, fuse)
        D_F2 = torch.multiply(w_2_1, fuse)
        F_1 = torch.add(D_F1, other=x1, alpha=1)
        F_2 = torch.add(D_F2, other=x2, alpha=1)

        return F_1, F_2

class Cross_layer(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0
    ):
        super().__init__()
        self.d_model = hidden_dim
        self.texture_enhance1 = Enhancement_texture_LDC(self.d_model,self.d_model)
        self.texture_enhance2 = Enhancement_texture_LDC(self.d_model, self.d_model)
        self.Diff_enhance = Differential_enhance(self.d_model)

    def forward(self, Fuse, x1,x2):
        TX_x1 = self.texture_enhance1(x1)
        TX_x2 = self.texture_enhance2(x2)

        DF_x1, DF_x2 = self.Diff_enhance(Fuse, x1,x2)
        F_1 = TX_x1 +DF_x1
        F_2 = TX_x2 +DF_x2

        return F_1, F_2
##########################################################################
##  Spatial-Frequency Feature Enhancement
##########################################################################
class SRGB(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dropout=0.,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        self.ssm1  = ES2D(
            d_model=self.d_inner,
            d_state=d_state
        )
        self.out_norm = LayerNorm2d(self.d_inner)

    def forward(self, x: torch.Tensor, **kwargs):
        B, C, H, W = x.shape
        #skip = x
        
        x = self.out_norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        out = self.ssm1(x)
        
       
        #out = out + skip
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

class FIR(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.select = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )
       
        self.ssm  = ES2D(
            d_model=dim,
            d_state=d_state
        )

       
        self.out_norm = LayerNorm2d(dim)
        
       
    def forward(self, x):  
        B,C,H,W = x.shape
        skip = x
        
        
        x = self.out_norm(x)
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Use float32 for FFT operations to avoid cuFFT restrictions on half precision
        original_dtype = x.dtype
        x = x.to(torch.float32)
        
        y = torch.fft.rfft2(x) + 1e-8
        a = torch.abs(y)
        p = torch.angle(y)
        
        # Convert back to original dtype for operations to match layer types
        a = a.to(original_dtype)
        a = self.select(a)
        
       
        p = p.permute(0, 2, 3, 1).contiguous()
        p = p.to(original_dtype)
        p = self.ssm(p)
        p = p.permute(0, 3, 1, 2).contiguous()
       
        # Ensure real and imag are in float32 for FFT operations
        real = a.to(torch.float32) * torch.cos(p.to(torch.float32))
        imag = a.to(torch.float32) * torch.sin(p.to(torch.float32))
        out = torch.complex(real, imag) + 1e-8
        
        # inverse FFT can handle complex inputs directly
        out = torch.fft.irfft2(out, s=(H, W), norm='backward') + 1e-8
        out = torch.abs(out)+1e-8
        out = out.to(original_dtype)
        

        # Ensure skip is same dtype as out
        skip = skip.to(original_dtype)
        
        out = out + skip

        return out

class dual_Vmamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.SPA = SRGB(d_model=d_model)
        self.FRE = FIR(dim=d_model)
        self.out_proj = nn.Linear(d_model, d_model)         
        self.d_model = d_model 

    def forward(self, x1, x2): 
        #perm = torch.randperm(self.d_model).to(x1.device) 
        skip = x1

        x1 = self.SPA(x1)
        x2 = self.FRE(x2)

        x2 = x1 + x2 
        x2 = self.out_proj(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x1 = x1 + skip                                                            
        return x1, x2

class dual_swapping_Vmamba(nn.Module): 
    """
    Build chnnel swapping Vmamba module
    input:x1,x2 with the shape of "B, C, H, W"
    output:x1,x2 with the shape of "B, C, H, W"
    """ 
    def __init__(
        self, 
        dim, 
        depth=1, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            dual_Vmamba(
                d_model=dim
            )
            for i in range(depth)])
        
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample1 = downsample(dim=dim, norm_layer=norm_layer)
            self.downsample2 = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample1 = None
            self.downsample2 = None
        
    def forward(self, x1, x2):
        #x1,x2 = x[0], x[1]
        #x1 = x1.permute(0, 2, 3, 1)
        #x2 = x2.permute(0, 2, 3, 1)
        #assert x1.shape == x2.shape, "Inputs must have the same shape."
        #chnnels = x1.shape[-1]

        #split_size = chnnels // 4
        #x1_part1, x1_part2, x1_part3, x1_part4 = torch.split(x1, split_size, dim=-1)
        #x2_part1, x2_part2, x2_part3, x2_part4 = torch.split(x2, split_size, dim=-1)

        #x1 = torch.cat([x1_part1, x2_part2, x1_part3, x2_part4],dim=-1)
        #x2 = torch.cat([x2_part1, x1_part2, x2_part3, x1_part4],dim=-1)

        #x1,x2 = x1.permute(0, 3, 1, 2).contiguous(), x2.permute(0, 3, 1, 2).contiguous()
        even_idx = torch.arange(x1.shape[1])%2 == 1
        x1_swap, x2_swap = x1.clone(), x2.clone()
        x1_swap[:,even_idx], x2_swap[:,even_idx] = x2[:,even_idx], x1[:,even_idx]
        x1, x2 = x1_swap, x2_swap

        
        for blk in self.blocks:
            if self.use_checkpoint:
                x1, x2 = checkpoint.checkpoint(blk, x1, x2)
            else:
                x1 = x1.cuda()
                x2 = x2.cuda()
                x1, x2 = blk(x1, x2)
        
        
        if self.downsample1 is not None:
            x1 = self.downsample1(x1)
        if self.downsample2 is not None:
            x2 = self.downsample2(x2)
        
        #x1 = x1.permute(0, 2, 3, 1)
        #x2 = x2.permute(0, 2, 3, 1)
        #chnnels = x1.shape[-1]

        #split_size = chnnels // 4
        #x1_part1, x1_part2, x1_part3, x1_part4 = torch.split(x1, split_size, dim=-1)
        #x2_part1, x2_part2, x2_part3, x2_part4 = torch.split(x2, split_size, dim=-1)

        #x1 = torch.cat([x1_part1, x2_part2, x1_part3, x2_part4],dim=-1)
        #x2 = torch.cat([x2_part1, x1_part2, x2_part3, x1_part4],dim=-1)

        #x1,x2 = x1.permute(0, 3, 1, 2).contiguous(), x2.permute(0, 3, 1, 2).contiguous()
        even_idx = torch.arange(x1.shape[1])%2 == 1
        x1_swap, x2_swap = x1.clone(), x2.clone()
        x1_swap[:,even_idx], x2_swap[:,even_idx] = x2[:,even_idx], x1[:,even_idx]
        x1, x2 = x1_swap, x2_swap


        return x1, x2

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
##########################################################################
##  Cross Mamba block
##########################################################################
class VSSBlock_Cross_new(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.Cross_layer = Cross_layer(hidden_dim)
        self.self_attention_cross = SS2D_cross(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.self_attention_cross_spatial = simam_module(channels=hidden_dim)
        self.dual_vmamba = dual_swapping_Vmamba(dim=hidden_dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, input1: torch.Tensor, input2:torch.Tensor):
        x_1 = input1.permute(0, 3, 1, 2) 
        x_2 = input2.permute(0, 3, 1, 2) 

        F_1, F_2 = self.dual_vmamba(x_1, x_2)

        F_1 = F_1.permute(0, 2, 3, 1)
        F_2 = F_2.permute(0, 2, 3, 1)
        Cross_x1x2 = self.self_attention_cross(self.ln_1(F_1),self.ln_2(F_2)) #(b, h, w, c)
        Cross_x1x2_ = Cross_x1x2.permute(0, 3, 1, 2) #(b, c, h, w )

        Cross_x1x2_spatial =  self.self_attention_cross_spatial(Cross_x1x2_) #(b, c, h, w)
        Cross_x1x2_spatial = Cross_x1x2_spatial.permute(0, 2, 3, 1)


        x = Cross_x1x2 + Cross_x1x2_spatial
        return x

##########################################################################
##  Multi-Modal Content Harmonizer(MMCH)
##########################################################################
class MMCH(nn.Module):
    def __init__(self, dim=32, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(MMCH, self).__init__()
        self.dim = dim
        self.cross_attention = VSSBlock_Cross_new(hidden_dim=dim, norm_layer=norm_layer ,d_state=dim//2)

    def forward(self, vi_b, ir_b):
        vi = vi_b
        ir = ir_b
        
        vi = vi.permute(0, 2, 3, 1)
        ir = ir.permute(0, 2, 3, 1)

        Ec = self.cross_attention(vi, ir).permute(0, 3, 1, 2)
    
        return Ec
    
if __name__ == '__main__':
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    height = 643
    width = 481       
    
    MMCH = MMCH().to(device)

    batch_size = 1
    channels = 32
   

    structure_I = torch.randn(batch_size, channels, height, width).to(device)
    structure_V = torch.randn(batch_size, channels, height, width).to(device)

    mask = torch.randn(batch_size, 1, height, width).to(device) 
    mask_V = torch.randn(batch_size, 1, height, width).to(device) 
    mask_I = torch.randn(batch_size, 1, height, width).to(device)
   
    
    print(f"input : {structure_I.shape}")
    print(f"input : {structure_V.shape}")
    
    
    MMCH.eval()
    with torch.no_grad():
            fused_structure = MMCH(structure_I, structure_V)

            print(f"fusion output: {fused_structure.shape}")