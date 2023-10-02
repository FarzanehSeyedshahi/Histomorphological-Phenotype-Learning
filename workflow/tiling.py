from tiatoolbox.tools.pyramid import TilePyramidGenerator
from tiatoolbox.wsicore.wsireader import WSIReader
import numpy as np
import imageio
import os

main_path = "/mnt/cephfs/home/users/fshahi/sharedscratch/Projects/Histomorphological-Phenotype-Learning"
wsi_path = "{}/datasets/TCGA_MESO/fd0a12b3-5e93-4106-94a4-3ebd418246ae".format(main_path)

# get a list of all the files in the directory
img_list = os.listdir(wsi_path)
img_list = [os.path.join(wsi_path, f) for f in img_list]
img_list = [f for f in img_list if f.endswith('.svs')]

print(img_list)


wsi = WSIReader.open(img_list[0])


tile_generator = TilePyramidGenerator(wsi=wsi,tile_size=1080)
print(tile_generator.level_count)

tile_0_0 = tile_generator.get_tile(level=1, x=0, y=0)
tile_0_0.save(os.path.join(os.getcwd(), "tile_0_0.png"))

location = (0, 0)
size = (256, 256)
# Read a region at level 0 (baseline / full resolution)
img = wsi.read_rect(location, size)
# Read a region at 0.5 microns per pixel (mpp)
img = wsi.read_rect(location, size, 0.5, "mpp")
# This could also be written more verbosely as follows
img = wsi.read_rect(location, size, resolution=(0.5, 0.5), units="mpp")
imageio.imwrite(os.path.join(os.getcwd(), "tile_0_0_0.5mpp.png"), img)

