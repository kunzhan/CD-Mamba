import os
import glob
from PIL import Image
import numpy as np


base_path = '/home/xtx/boime/bigfutrue'
bluepng_path = os.path.join(base_path, 'bluepng')
greenpng_path = os.path.join(base_path, 'greenpng')
redpng_path = os.path.join(base_path, 'redpng')
output_path = os.path.join(base_path, 'test123')

if not os.path.exists(output_path):
    os.makedirs(output_path)


pattern = os.path.join(bluepng_path, 'LC8*.png')
matching_files = glob.glob(pattern)

for file_path in matching_files:

    file_name = os.path.basename(file_path)
    prefix = file_name[:21] 


    green_band_path = os.path.join(greenpng_path, prefix + '_B3.png')
    blue_band_path = file_path
    red_band_path = os.path.join(redpng_path, prefix + '_B4.png')
    print(red_band_path)
    if not (os.path.exists(green_band_path) and os.path.exists(blue_band_path) and os.path.exists(red_band_path)):
        print(f"Missing files for prefix {prefix}")
        continue


    green_band = np.array(Image.open(green_band_path).convert('I;16'))
    blue_band = np.array(Image.open(blue_band_path).convert('I;16'))
    red_band = np.array(Image.open(red_band_path).convert('I;16'))


    red_band = (red_band.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    green_band = (green_band.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    blue_band = (blue_band.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

    red_band_img = Image.fromarray(red_band)
    green_band_img = Image.fromarray(green_band)
    blue_band_img = Image.fromarray(blue_band)


    assert red_band_img.size == green_band_img.size == blue_band_img.size, f"Images for prefix {prefix} do not have the same dimensions"


    rgb_image = Image.merge('RGB', (red_band_img, green_band_img, blue_band_img))


    output_file_path = os.path.join(output_path, prefix + '.png')
    rgb_image.save(output_file_path)

    print(f"Processed and saved {output_file_path}")