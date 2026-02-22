import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
    def forward(self,pred,label):
        edge_loss=F.binary_cross_entropy(pred,label) 
        return edge_loss

class SSIM_SLoss(nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, C2=0.0009, device='cuda', eps=1e-8):
        super().__init__()
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.C2 = C2
        self.device = device
        self.eps = eps

    def ssim_struct(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
            x = x.unsqueeze(0).unsqueeze(0) if len(x.shape) == 2 else x.unsqueeze(0)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float().to(self.device)
            y = y.unsqueeze(0).unsqueeze(0) if len(y.shape) == 2 else y.unsqueeze(0)
        x = x.unsqueeze(0) if len(x.shape) == 3 else x
        y = y.unsqueeze(0) if len(y.shape) == 3 else y
        b, c, h, w = x.shape

        gauss = torch.arange(self.win_size, dtype=torch.float32, device=self.device) - self.win_size//2
        gauss = torch.exp(-gauss**2 / (2 * self.win_sigma**2))
        gauss = gauss / (gauss.sum() + self.eps)
        win = (gauss.unsqueeze(1) * gauss.unsqueeze(0)).unsqueeze(0).repeat(c, 1, 1, 1)
        win = win.to(x.dtype)

        pad = self.win_size // 2
        mu_x = F.conv2d(x, win, padding=pad, groups=c)
        mu_y = F.conv2d(y, win, padding=pad, groups=c)

        sigma_x2 = F.conv2d(x*x, win, padding=pad, groups=c) - mu_x**2
        sigma_y2 = F.conv2d(y*y, win, padding=pad, groups=c) - mu_y**2
        sigma_x2 = torch.clamp(sigma_x2, min=self.eps)
        sigma_y2 = torch.clamp(sigma_y2, min=self.eps)

        sigma_x = torch.sqrt(sigma_x2 + self.eps)
        sigma_y = torch.sqrt(sigma_y2 + self.eps)

        sigma_xy = F.conv2d(x*y, win, padding=pad, groups=c) - mu_x*mu_y
        sigma_xy = torch.clamp(sigma_xy, min=-1e3, max=1e3)

        C3 = self.C2 / 2
        struct = (sigma_xy + C3) / (sigma_x * sigma_y + C3 + self.eps)
        struct = torch.clamp(struct, 0.0, 1.0)
        return struct.squeeze()

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=-1e3, max=1e3)
        target = torch.clamp(target, min=-1e3, max=1e3)
        
        struct = self.ssim_struct(pred, target)
        loss = 1 - torch.mean(struct)
        loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.tensor(0.0, device=loss.device), loss)
        return loss



