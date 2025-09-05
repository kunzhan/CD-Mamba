import cv2
import numpy as np
import os
import re 
pic_path = '/home/xtx/boime/bigfutrue/test123/rgb/' # Path to the small images
pic_target = '/home/xtx/boime/big2/' # Path to save the reconstructed large image from patches
if not os.path.exists(pic_target):
    os.mkdir(pic_target)
pic_path1 = "/home/xtx/boime/bigfutrue/test123/rgb/" 

large_picture_names = os.listdir(pic_path1)
# print(large_picture_names)
# large_picture_names = [large_picture_names[-44:] for large_picture_names in large_picture_names] 
truncated_picture_names = []  
for name in large_picture_names:  
    parts = name.split('_')  # Split the filename into segments usin _
    if len(parts) > 5: # If there are at least 6 segments
        truncated_name = '_'.join(parts[5:])   # Keep and join all segments from the 6th onward
    else:  
        # If the filename has fewer than 6 segments, keep it unchanged
        truncated_name = name  
    truncated_picture_names.append(truncated_name)  
  

large_picture_names = set(truncated_picture_names) 

picture_names = os.listdir(pic_path)                 

(width, length, depth) = (384,384,3)
if len(picture_names)==0:
    print("none")
else:
    num = 0
    for pic in large_picture_names:
        # List all files in a folder
        txt_file_path = '/home/xtx/boime/test/patch_in_123.txt'  

        filenames = []  
        
 
        with open(txt_file_path, 'r') as file:  
            for line in file:  
                filenames.append(line.strip())  
        
        filenames = os.listdir(pic_path)  
        
        matching_filenames = [filename for filename in filenames if pic in filename]
        a = 0 
        b = 0
        for filename in matching_filenames:
            w = int(filename.split("_")[2])
            l = int(filename.split("_")[4])
            if w > a:
                a = w 
            if l > b:
                b = l  
        num_width = a
        num_length= b
        num += 1
        
        splicing_pic = np.zeros((num_width*width, num_length*length, depth))

        for idx in range(0, 1):
            k = 0
            splicing_pic = np.zeros((num_width*width, num_length*length, depth))
            for i in range(1, num_width+1):
                for j in range(1, num_length+1):
                        k += 1
                        img_part = cv2.imread(pic_path + 'patch_{}_{}_by_{}_'.format(k, i, j)+pic,1)         
                        splicing_pic[ width*(i-1) : width*i, length*(j-1) : length*j, :] = img_part
            cv2.imwrite(pic_target + pic, splicing_pic)
    print(num)
    print("done!!!")