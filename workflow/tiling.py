import argparse
from aicsimageio import AICSImage
import datetime
import numpy as np
import os
from PIL import Image

parser = argparse.ArgumentParser(description='Tile input images to a specified size.')
parser.add_argument("--input", type=str, default='', help='Path to images')
parser.add_argument("--output", type=str, default='', help='Directory to write tiles to')
parser.add_argument("--tile_size", type=int, default=224, help='Tile dimensions in pixels')
parser.add_argument("--pixel_size", type=float, default='', help='Pixel size to generate tiles at')
parser.add_argument("--magnification", type=str, default='40.0', help='Magnification of the image, just for subdirectory naming to comply with DeepPATH convention')
parser.add_argument("--background", type=int, default=40, help='Maximum background tolerated')
parser.add_argument("--tma", type=bool, default=False, help='Whether the image being tiled is a TMA core')
args = parser.parse_args()

output_path = args.output
input_path = args.input
tile_size = args.tile_size
pixel_size = args.pixel_size
magnification = args.magnification
background = args.background
tma_flag = args.tma

print ("Start time :", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def tile_extractor(image, tile_size, pixel_size, magnification, background, output, format):
    tiled_path = os.path.join(output, os.path.basename(image).split(format)[0] + '_files', magnification)
    if not os.path.exists(tiled_path):
        os.makedirs(tiled_path)
    # if os.path.exists(tiled_path):
    #     print(f'Path exists: {tiled_path}')
    #     return
    img = AICSImage(image, dask_tiles=True)
    print(f'Processing: {os.path.basename(image)}')
    print(f'Image Shape: {img.shape}')
    lazy_t0 = img.get_image_dask_data("YXS")  
    img_array = lazy_t0.compute()
    img_x = img_array.shape[0]
    img_y = img_array.shape[1]
    mpp = img.physical_pixel_sizes[1]
    # mpp = 0.4415
    scale = pixel_size / mpp
    print(f'Physical Pixel Size: {mpp}')
    print(f'Requested Pixel Size: {pixel_size}')
    print(f'Scale: {scale}')
    print(f'Scaled Tile: {int(np.round(tile_size * scale))}')
    scaled_tile = int(np.round(tile_size * scale))
    tiles_range_x = int(np.floor(img_x / scaled_tile))
    tiles_range_y = int(np.floor(img_y / scaled_tile))
    for i in range(0, tiles_range_x + 1):
        for j in range(0, tiles_range_y + 1):
            tile = img_array[j * scaled_tile:(j+1) * scaled_tile, i * scaled_tile:(i+1) * scaled_tile, :] 
            tile_filename = f'{i}_{j}.jpeg'
            savetile_path = os.path.join(tiled_path, tile_filename)
            if tile.shape[0] == scaled_tile and tile.shape[1] == scaled_tile:
                jpeg = Image.fromarray(tile, mode='RGB').resize((tile_size,tile_size))
                gray = jpeg.convert('L')
                bw = gray.point(lambda x: 0 if x < 220 else 1, 'F')
                bkg = np.average(bw)
                if bkg <= (background / 100):
                    jpeg.save(savetile_path, quality=90)
                else:
                    continue
            else:
                continue

def tile_extractor_tma(image, tile_size, pixel_size, magnification, background, output, format):
    tiled_path = os.path.join(output, os.path.basename(image).split(format)[0] + '_files', magnification)
    if not os.path.exists(tiled_path):
        os.makedirs(tiled_path)
    img = AICSImage(image, dask_tiles=True)
    print(f'Processing: {os.path.basename(image)}')
    lazy_t0 = img.get_image_dask_data("YXS")  
    img_array = lazy_t0.compute()
    img_x = img_array.shape[0]
    img_y = img_array.shape[1]
    mpp = img.physical_pixel_sizes[1]
    scale = pixel_size / mpp
    scaled_tile = int(np.round(tile_size * scale))
    tiles_range_x = int(np.floor(img_x / scaled_tile))
    tiles_range_y = int(np.floor(img_y / scaled_tile))
    core_centroid_x = img_x // 2
    core_centroid_y = img_y // 2
    upper_left = img_array[core_centroid_y-scaled_tile:core_centroid_y, core_centroid_x-scaled_tile:core_centroid_x]
    lower_left = img_array[core_centroid_y:core_centroid_y+scaled_tile, core_centroid_x-scaled_tile:core_centroid_x]
    upper_right = img_array[core_centroid_y-scaled_tile:core_centroid_y, core_centroid_x:core_centroid_x+scaled_tile]
    lower_right = img_array[core_centroid_y:core_centroid_y+scaled_tile, core_centroid_x:core_centroid_x+scaled_tile]
    tiles = [upper_left, lower_left, upper_right, lower_right]
    filenames = ['0_0.jpeg', '0_1.jpeg', '1_0.jpeg', '1_1.jpeg']
    for i, tile in enumerate(tiles):
        tile_filename = filenames[i]
        savetile_path = os.path.join(tiled_path, tile_filename)
        jpeg = Image.fromarray(tile, mode='RGB').resize((tile_size,tile_size))
        jpeg.save(savetile_path, quality=90)

# Function to calculate the centroid of an image and draw four tiles around to capture the tissue in the middle and little to no white space
# Written specifically for low magnificantion -- todo: calculate bounds automatically and iterate from middle rather than top-left (for higher magnification)
if tma_flag == True:
    for file in os.scandir(input_path):
        if file.name.endswith('.tif'):
            image = file.path
            tile_extractor_tma(image=image, tile_size=tile_size, pixel_size=pixel_size, magnification=magnification, background=100, output=output_path, format='.tif')
else:
    if input_path.endswith('.ndpi'):
        image = input_path
        tile_extractor(image=image, tile_size=tile_size, pixel_size=pixel_size, magnification=magnification, background=background, output=output_path, format='.ndpi')
    elif input_path.endswith('.tif'):
        image = input_path
        tile_extractor(image=image, tile_size=tile_size, pixel_size=pixel_size, magnification=magnification, background=background, output=output_path, format='.tif')
    elif input_path.endswith('.svs'):
        image = input_path
        tile_extractor(image=image, tile_size=tile_size, pixel_size=pixel_size, magnification=magnification, background=background, output=output_path, format='.svs')

    elif input_path.endswith('.jpg'):
        image = input_path
        tile_extractor(image=image, tile_size=tile_size, pixel_size=pixel_size, magnification=magnification, background=background, output=output_path, format='.jpg')
    else:
        print(input_path)
        print('Invalid image file passed')

# for file in os.scandir(input_path):
#     if file.name.endswith('.ndpi'):
#         image = file.path
#         tile_extractor(image=image, tile_size=tile_size, pixel_size=pixel_size, magnification=magnification, background=background, output=output_path, format='.ndpi')
        
print ("End time :", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# python tiling.py --input /data/images --output /data/tiles --tile_size 224 --pixel_size 0.4415 --magnification 40.0 --background 40 --tma False