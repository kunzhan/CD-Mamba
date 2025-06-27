import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch  
# from models.unet import UNet
# from models.vision_transformer import SwinUnet 
from models.cloud import cdMamba
# from models.rs_mamba_ss import RSM_SS
# from a_swin.attention_swin_unet import SwinAttentionUnet
# from models.vmunet import VMUNet
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"
from skimage.io import imread
from utils import *
from configs.config_setting import setting_config
from dataprepare.tifsemi import SemiDataset
import warnings
import numpy as np
from PIL import Image 
warnings.filterwarnings("ignore")



def main(config):
    rot = './test'
    name = 'boime'
    sys.path.append(rot+ '/'+name)

    resume_model = os.path.join("./results/cloudmamba/checkpoints/latest.pth")
    outputs = os.path.join(rot+ '/'+name)

    if not os.path.exists(outputs):
        os.makedirs(outputs)

    set_seed(config.seed)
    gpu_ids = [0]
    torch.cuda.empty_cache()
    


    print('#----------Prepareing Models----------#') 
    model = cdMamba(num_classes=1,input_channels=4)
    # model = SwinUnet(num_classes=1,img_size=384)
    # model = RSM_SS()
    # model = SwinAttentionUnet()
    # model = UNet(num_classes=1,in_channels=4)
    # model = VMUNet(input_channels=4, num_classes=1)
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('no')
    print('#----------Testing----------#')

    to_test = {'test':'/data/boime'}
    val_path = '/data/boime/test/patch_in_456.txt'

    with torch.no_grad():
        TN = 0
        FP = 0
        FN = 0
        TP = 0
        for name, root in to_test.items():

            # 获取图片名称list,txt
            with open(val_path, 'r') as f:
                ids = f.read().splitlines()
            img_list = ids
            
            # 获取图片名称list,遍历文件夹
            # img_list = [os.path.splitext(f)[0] for f in os.listdir(root+'/mask') if f.endswith('.png')]

            for idx, img_name in enumerate(img_list):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                image_red = os.path.join(root+'/red/', img_name+'.TIF') 
                image_green = os.path.join(root+ '/green/', img_name+'.TIF') 
                image_blue = os.path.join(root+ '/blue/', img_name+'.TIF') 
                image_nir = os.path.join(root+ '/nir/',  img_name+'.TIF') 
                image_gt = os.path.join(root+'/mask/',img_name+'.png')

                image_red = imread(image_red)
                image_green = imread(image_green)
                image_blue = imread(image_blue)
                image_nir = imread(image_nir)
                image_red = image_red.astype(np.float32)
                image_green = image_green.astype(np.float32)
                image_blue = image_blue.astype(np.float32)
                image_nir = image_nir.astype(np.float32)
                mask = imread(image_gt)/255 
                img = np.stack((image_red, image_green, image_blue, image_nir), axis=0)
                img = np.expand_dims(img, axis=0)

                img = torch.from_numpy(img).float().cuda(non_blocking=True)  

                output = model(img).cuda()  

                out = output.squeeze(1).cpu().detach().numpy()

                output = np.where(np.squeeze(out, axis=0) > 0.5,1, 0) 


                preds = np.array(out).reshape(-1)
                gts = np.array(mask).reshape(-1)

                y_pre = np.where(preds>=0.5, 1, 0)
                y_true = np.where(gts>=0.5, 1, 0)
                tp = (y_pre * y_true).sum()
                tn = ((1 - y_pre) * (1 - y_true)).sum() 
                fp = (y_pre * (1 - y_true)).sum() 
                fn = ((1 - y_pre) * y_true).sum()
                
                TN += tn
                FP += fp
                FN += fn
                TP += tp
                if output.dtype != np.uint8:  
                    output = (output * 255).astype(np.uint8) 
                  
                img = Image.fromarray(output, 'L')  
                img.save(outputs + '/' + img_name + '.png')



    accuracy = float(TN + TP) / float(TN+ TP + FP + FN)
    f1_or_dsc = float(2 * TP) /  (2 * TP + FP + FN) 
    miou = float(TP) / float(TP + FP + FN)
    print(accuracy,f1_or_dsc,miou)



if __name__ == '__main__':
    config = setting_config
    main(config)