import scipy.io as sio
import scipy.misc as sc
from copy import deepcopy
import math
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from PIL import Image
from skimage.io import imread
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import yaml
import argparse
from tqdm import tqdm
class SemiDataset(Dataset):
    def __init__(self, root, mode, id_path=None, nsample=None):
        self.root = root
        self.mode = mode

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
            # with open(mini_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
               
    

    def __getitem__(self, item):
        id = self.ids[item]
        image_red = os.path.join(self.root, 'red', id+'.TIF') 
        image_green = os.path.join(self.root, 'green', id+'.TIF') 
        image_blue = os.path.join(self.root, 'blue', id+'.TIF') 
        image_nir = os.path.join(self.root, 'nir',  id+'.TIF') 
        image_gt = os.path.join(self.root,'mask',id+'.png')

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


        if self.mode == 'train_l':
            return img, mask
        
        if self.mode == 'val':
            return img, mask

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        def augment_data(image):
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1).copy()  # 水平翻转并复制数组
            # 随机亮度调整
            brightness_factor = np.random.uniform(0.5, 1.5)  # 随机亮度因子
            image = np.clip(image * brightness_factor, 0, 255)

            # 随机对比度调整
            contrast_factor = np.random.uniform(0.8, 1.2)  # 随机对比度因子
            mean = np.mean(image, axis=(1, 2), keepdims=True)
            image = np.clip((image - mean) * contrast_factor + mean, 0, 255)
            return image

        img_s1 = augment_data(img_s1)
        # img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = augment_data(img_s2)


        mask = torch.from_numpy(np.array(mask)).long()
        # ignore_mask[mask == 254] = 255
        return img_w, img_s1, img_s2# , ignore_mask, cutmix_box1, cutmix_box2
    def __len__(self):
        return len(self.ids)
    
if __name__ == '__main__':
    trainset_l = SemiDataset('/home/xtx/boime', 'train_l', '/home/xtx/boime/train/patch_in_123.txt')
    train_loader = DataLoader(trainset_l, batch_size=1,
                               pin_memory=False, num_workers=0, drop_last=True, sampler=None)
    for i, (img_w,mask) in enumerate(train_loader):
            mask = mask.cuda()
            img_w = img_w.cuda()
            # print(mask.dtyp
            # print(mask.shape)
            print(mask.shape)
