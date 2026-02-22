# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from Encoder_Decoder import HFE, DAFD, EdgeDecoder, GL_DAC
from MMCH import MMCH
from IMDE import IMDE
from LABTEE import LearnableCanny
from utils.dataset import H5Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, EdgeLoss, SSIM_SLoss
import kornia
import logging


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
criteria_edge = EdgeLoss()
criteria_SSIM_s = SSIM_SLoss()
model_str = 'EGA-CMTNet'

# . Set the hyper-parameters for training
num_epochs = 200   # total epoch
phase2_epochs = 40 #
phase1_epochs = 20 

 
lr = 1e-3
weight_decay = 1e-5
batch_size = 8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']

logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)


log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# Coefficients of the loss function
coeff_ssim_s = 0.5
coeff_mse_loss_VF = 0.5 
coeff_mse_loss_IF = 0.5

coeff_tv = 0.5
coeff_edge = 0.5

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5



# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Encoder = nn.DataParallel(HFE()).to(device)
Decoder = nn.DataParallel(DAFD()).to(device)
EdgeDecoder = nn.DataParallel(EdgeDecoder()).to(device)
MMCH = nn.DataParallel(MMCH()).to(device)
IMDE = nn.DataParallel(IMDE()).to(device)
GL_DAC = nn.DataParallel(GL_DAC()).to(device)
canny = nn.DataParallel(LearnableCanny()).to(device)


# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    Encoder.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer2 = torch.optim.Adam(
    Decoder.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer3 = torch.optim.Adam(
    MMCH.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer4 = torch.optim.Adam(
    IMDE.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer5 = torch.optim.Adam(
    canny.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer6 = torch.optim.Adam(
    EdgeDecoder.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer7 = torch.optim.Adam(
    GL_DAC.parameters(), lr=lr, weight_decay=weight_decay
)



scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)
scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=optim_step, gamma=optim_gamma)
scheduler7 = torch.optim.lr_scheduler.StepLR(optimizer7, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')



# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0,  
                         pin_memory=True)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

log_file = os.path.join(logs_dir, f'{model_str}_{timestamp}.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(log_format)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(log_format)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    step = 0
    
    torch.backends.cudnn.benchmark = True
    prev_time = time.time()

    for epoch in range(num_epochs):
       ''' train '''
       for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        Encoder.train()
        Decoder.train()
        MMCH.train()
        IMDE.train()
        EdgeDecoder.train()
        canny.train()
        GL_DAC.train()
        
        
        Encoder.zero_grad()
        Decoder.zero_grad()
        MMCH.zero_grad()
        IMDE.zero_grad()
        EdgeDecoder.zero_grad()
        canny.zero_grad()
        GL_DAC.zero_grad()
        

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()
        optimizer6.zero_grad()
        optimizer7.zero_grad()

        if epoch < phase1_epochs : 
            _,mask_V,mask_I = canny(data_VIS, data_IR)
            ssim_I = criteria_SSIM_s(data_VIS, mask_V)
            ssim_V = criteria_SSIM_s(data_IR, mask_I)
            loss_ssim = ssim_I + ssim_V
            loss = coeff_ssim_s*loss_ssim 

            loss.backward()

            nn.utils.clip_grad_norm_(
                canny.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer5.step()
        elif epoch < phase1_epochs + phase2_epochs and epoch >= phase1_epochs: #Phase II
            
            for param in canny.parameters():
                param.requires_grad = False
            
            _,mask_V,mask_I = canny(data_VIS, data_IR)

            feature_V_B, feature_V_D = Encoder(data_VIS)
            feature_I_B, feature_I_D = Encoder(data_IR)

            feature_V_dif,feature_V_acc = GL_DAC(feature_V_B, feature_V_D, mask_V)
            feature_I_dif,feature_I_acc = GL_DAC(feature_I_B, feature_I_D, mask_I)

            data_VIS_hat, feature_VIS_edge = Decoder(data_VIS, feature_V_dif, feature_V_acc)
            data_IR_hat, feature_IR_edge = Decoder(data_IR, feature_I_dif, feature_I_acc)

            pred_edge_V = EdgeDecoder(feature_VIS_edge)
            pred_edge_I = EdgeDecoder(feature_IR_edge)

            edge_loss_V = criteria_edge(pred_edge_V, mask_V)
            edge_loss_I = criteria_edge(pred_edge_I, mask_I)

            loss_edge = edge_loss_V + edge_loss_I

            mse_loss_V = Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss_V = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            Gradient_loss_I = L1Loss(kornia.filters.SpatialGradient()(data_IR),
                                   kornia.filters.SpatialGradient()(data_IR_hat))
            Gradient_loss = Gradient_loss_V + Gradient_loss_I

            
            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_tv * Gradient_loss + coeff_edge * loss_edge

            loss.backward()
            nn.utils.clip_grad_norm_(
                Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                GL_DAC.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                EdgeDecoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            
            optimizer1.step()  
            optimizer2.step()
            optimizer6.step()
            optimizer7.step()

        else:
            for param in canny.parameters():
                param.requires_grad = False  
            mask,_,_ = canny(data_VIS, data_IR)
            feature_V_B, feature_V_D = Encoder(data_VIS)
            feature_I_B, feature_I_D = Encoder(data_IR)

            feature_F_B = MMCH(feature_V_B, feature_I_B)  
            feature_F_D = IMDE(feature_V_D, feature_I_D)

            feature_F_dif,feature_F_acc = GL_DAC(feature_F_B, feature_F_D, mask)

            data_Fuse, feature_F = Decoder(data_VIS, feature_F_dif, feature_F_acc)

            pred_edge = EdgeDecoder(feature_F)

            edge_loss = criteria_edge(pred_edge, mask)

            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse)
            
            loss =  fusionloss + edge_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                MMCH.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                IMDE.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                GL_DAC.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                EdgeDecoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)   
           
            optimizer1.step()  
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer6.step()
            optimizer7.step()
          

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %10f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate
    if epoch >= phase1_epochs + phase2_epochs:
        scheduler1.step()  
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()
        scheduler6.step()
        scheduler7.step()
    elif epoch < phase1_epochs:
        scheduler5.step()
    else:
        scheduler1.step()  
        scheduler2.step()
        scheduler6.step()
        scheduler7.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    if optimizer5.param_groups[0]['lr'] <= 1e-6:
        optimizer5.param_groups[0]['lr'] = 1e-6
    if optimizer6.param_groups[0]['lr'] <= 1e-6:
        optimizer6.param_groups[0]['lr'] = 1e-6
    if optimizer7.param_groups[0]['lr'] <= 1e-6:
        optimizer7.param_groups[0]['lr'] = 1e-6
    
    
    if True:
     checkpoint = {
        'Encoder': Encoder.state_dict(),
        'Decoder': Decoder.state_dict(),
        'MMCH': MMCH.state_dict(),
        'IMDE': IMDE.state_dict(),     
        'canny': canny.state_dict(),
        'EdgeDecoder': EdgeDecoder.state_dict(),
        'GL_DAC': GL_DAC.state_dict()
    }
    torch.save(checkpoint, os.path.join("models/EGA-CMTNet_"+timestamp+'.pth'))
