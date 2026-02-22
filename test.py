from Encoder_Decoder import HFE, DAFD, GL_DAC
from MMCH import MMCH
from IMDE import IMDE
from LABTEE import LearnableCanny
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models/EGA-CMTNet.pth"
for dataset_name in ["RoadScene"]:
    print("\n"*2+"="*80)
    model_name="EGA-CMTNet"
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name) 
    test_out_folder=os.path.join('test_result',dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(HFE()).to(device)
    Decoder = nn.DataParallel(DAFD()).to(device)
    MMCH = nn.DataParallel(MMCH()).to(device)
    IMDE = nn.DataParallel(IMDE()).to(device)
    GL_DAC = nn.DataParallel(GL_DAC()).to(device)
    canny = nn.DataParallel(LearnableCanny()).to(device)
   
    
    Encoder.load_state_dict(torch.load(ckpt_path)['Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['Decoder'])
    MMCH.load_state_dict(torch.load(ckpt_path)['MMCH'])
    IMDE.load_state_dict(torch.load(ckpt_path)['IMDE'])
    GL_DAC.load_state_dict(torch.load(ckpt_path)['GL_DAC'])
    canny.load_state_dict(torch.load(ckpt_path)['canny'])
    
    
    Encoder.eval()
    Decoder.eval()
    MMCH.eval()
    IMDE.eval()
    GL_DAC.eval()
    canny.eval()
    

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):
     
            data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS_YCbCr = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='YCrCb')[np.newaxis,np.newaxis, ...]/255.0
            


            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS_YCbCr = torch.FloatTensor(data_VIS_YCbCr)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
            data_VIS_YCbCr = data_VIS_YCbCr.cuda()
            data_VIS_Y = data_VIS_YCbCr[:, :, :, :, 0]
            data_VIS_Cr = data_VIS_YCbCr[:, :, :, :, 1]
            data_VIS_Cb = data_VIS_YCbCr[:, :, :, :, 2] 
            

            mask,_,_ = canny(data_VIS, data_IR)
           
            feature_V_B, feature_V_D = Encoder(data_VIS)
            feature_I_B, feature_I_D = Encoder(data_IR)

            feature_F_B = MMCH(feature_V_B, feature_I_B)
            feature_F_D = IMDE(feature_V_D, feature_I_D)

            feature_F_dif,feature_F_acc = GL_DAC(feature_F_B, feature_F_D, mask)

            data_Fuse, _ = Decoder(data_VIS, feature_F_dif, feature_F_acc)
            
            
            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            data_Fuse = data_Fuse.unsqueeze(-1)
            data_VIS_Cb = data_VIS_Cb.unsqueeze(-1)  
            data_VIS_Cr = data_VIS_Cr.unsqueeze(-1) 

            fi = torch.cat([data_Fuse, data_VIS_Cb, data_VIS_Cr], dim=-1)

            fi = np.squeeze((fi * 255).cpu().numpy())

            fi = cv2.cvtColor(fi, cv2.COLOR_YCrCb2BGR)

            img_save(fi, img_name.split(sep='.')[0], test_out_folder)

    
    eval_folder=test_out_folder  
    ori_img_folder=test_folder
    
    
    metric_result = np.zeros((11))
    for img_name in os.listdir(os.path.join(ori_img_folder,"ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi),Evaluator.AG(fi), Evaluator.MI(fi, ir, vi)
                                        , Evaluator.FMI_pixel(fi, ir, vi)
                                        , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                        , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                                        , Evaluator.PSNR(fi, ir, vi)])

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t SD\t SF\t AG\t MI\t FMI\tSCD\tVIF\tQabf\tSSIM\tPSNR")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))+'\t'
            +str(np.round(metric_result[4], 2))+'\t'
            +str(np.round(metric_result[5], 2))+'\t'
            +str(np.round(metric_result[6], 2))+'\t'
            +str(np.round(metric_result[7], 2))+'\t'
            +str(np.round(metric_result[8], 2))+'\t'
            +str(np.round(metric_result[9], 2))+'\t'
            +str(np.round(metric_result[10], 2))
            )
    print("="*80)
    
    