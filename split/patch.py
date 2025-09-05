# -*- coding:utf-8 -*-
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import glob


def crop_one_picture(path,filename,cols,rows):
    img = cv2.imread(path+filename,-1)# img
    img= cv2.copyMakeBorder(img, top=167, bottom=168, left=61, right=62,
                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]) # 21    
    sum_rows=img.shape[0]   
    sum_cols=img.shape[1]   
    save_path=path+"/crop{0}_{1}/".format(cols,rows) 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("The image was divided into a grid of {0} × {1} patches.".format(int(sum_cols/cols),int(sum_rows/rows))) 
    k = 0
    for i in range(int(sum_rows/rows)):
        for j in range(int(sum_cols/cols)):
            k += 1
            cv2.imwrite(save_path+'patch_{}_{}_by_{}_'.format(k, i+1, j+1)+ filename.split('.')[0] + \
            os.path.splitext(filename)[1],img[i*cols:(i+1)*cols,j*rows:(j+1)*rows]) # mask
    print("divided into {0} patches.".format(int(sum_cols/cols)*int(sum_rows/rows)))
    print("save to the path of {0}".format(save_path))
    
if __name__ == '__main__':

    path='/home/xtx/boime/bigfutrue/test123/'   
    picture_names = sorted(glob.glob(path + '*.png'))
    num = 0
    for picture_name in picture_names:
        print(picture_name)
        name = picture_name.split('/')[6] 
        print(name)
        crop_one_picture(path,name,384,384)