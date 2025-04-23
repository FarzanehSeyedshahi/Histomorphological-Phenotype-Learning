
import h5py
import glob
import os
import numpy as np
import cv2
from mpi4py import MPI
from itertools import chain
import argparse
import random

# srun -p cpu_short -c 11 -n 11 --mem-per-cpu=4G -t 00-02:00:00  --pty bash

# module unload gcc
# module load openmpi
# module load python/cpu/3.6.5

# mpiexec -n 10 python parallel_hdf5.py


parser = argparse.ArgumentParser(description="Convert jpeg images sorted is subfolders (1 per class) to hdf5 format.")

parser.add_argument("--input_path", type=str, default='',
                        help="input path (parent directory)")
parser.add_argument("--output", type=str, default='path/output.d5',
                        help="path and name of output")
# parser.add_argument("--chunks", type=int, default=40,
                        # help="must equal the number of cores")
parser.add_argument("--sub_chunks", type=int, default=20,
                        help="number of sub-chunks")
parser.add_argument("--wSize", type=int, default=224,
                        help="output window size")
# parser.add_argument("--mode", type=int, default=1,
                        # help="0 - path=tiled images in original folder; 1 - path=tiled images sorted in subfolders as labels; 2 - path=sorted images path")
parser.add_argument("--subset", type=str, default='train',
                        help="in mode 2 only, will only take tiles within a certain subset (train, test or valid; or combined) - in mode 1, it will be appened to the tags name within h5")
# parser.add_argument("--subsetout", type=str, default='None',
                        # help="in mode 2 only; mode of output h5 - same as subset if not specified")
# parser.add_argument("--startS", type=int, default=0,
                        # help="First image in the image list to take")
# parser.add_argument("--stepS", type=int, default=1,
                        # help="set to >1 if sample is needed. Every stepS image will be taken starting with image startS")
# parser.add_argument("--maxIm", type=int, default=-1,
#                         help="maximum number of images to take. All if set to -1")
parser.add_argument("--mag", type=float,
                        help="magnification to use (mode 0 and 1 only)")
parser.add_argument("--label", type=str, default='',
                        help="label to use (mode 0 only); either a string or a filepath with labels")
# parser.add_argument("--slideID", type=int, default='23',
#                         help="number of characters to use for the slide ID")
# parser.add_argument("--sampleID", type=int, default='14',
#                         help="number of characters to use for the sample (or patient) ID")






args = parser.parse_args()

# chunks = args.chunks
sub_chunks = args.sub_chunks
top_level = args.input_path
# if args.subsetout == 'None':
subsetout = args.subset
# else:
# 	subsetout = args.subsetout



WIDTH = args.wSize
HEIGHT = args.wSize

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
print("Process %d of %d on %s.\n" % (rank, size, name))

# get length of all images to add to hdf5
Images = []
# patterns = [f.name for f in os.scandir(args.input_path) if f.is_dir()]
# patterns1.sort()
# patterns = patterns1
slides = [f.name for f in os.scandir(top_level) if f.is_dir()]
for slide in slides:
    images = glob.glob(os.path.join(top_level, slide, str(args.mag) + "/*.jpeg"))
    Images.append(images)


# length of all images
ImageList = list(chain.from_iterable(Images))
# finish the code here

# print("aaa1 - " + str(len(ImageList1)))
# if args.startS > len(ImageList1):
#     args.startS = 0
#     print("startS is larger than the total number of images")
# if args.startS < 0:
#     args.startS = 0
# if args.stepS < 1:
#     args.stepS = 1
# Indx = [x for x in range(0, len(ImageList1), 1)]
# print("aaa3 - " + str(len(Indx)))
# ImageList = [ImageList1[x] for x in Indx]
# print("aaa3 - " + str(len(ImageList)))
# if args.maxIm > 0:
#     ImageList = ImageList[0:args.maxIm]
# print("aaa3 - " + str(len(ImageList)))
# ImageList = Images2

# patterns1.append('unknown')
# create hdf5 dataset
f = h5py.File(args.output, 'w')
# dset = f.create_dataset(subsetout+'_img', (len(ImageList), WIDTH, HEIGHT, 8), dtype='uint8') # for MIF
dset = f.create_dataset(subsetout+'_img', (len(ImageList), WIDTH, HEIGHT, 3), dtype='uint8') # for RGB
dset2 = f.create_dataset(subsetout+'_patterns', (len(ImageList), ),
                         dtype = 'S37')
dset3 = f.create_dataset(subsetout+'_slides', (len(ImageList), ),
                         dtype = 'S23')
dset4 = f.create_dataset(subsetout+'_tiles', (len(ImageList), ),
                         dtype = 'S16')
# dset5 = f.create_dataset(subsetout+'_labels', (len(ImageList), ))
dset6 = f.create_dataset(subsetout+'_hist_subtype', (len(ImageList), ),
                         dtype = 'S37')
dset7 = f.create_dataset(subsetout+'_samples', (len(ImageList), ),
                         dtype = 'S23')


# size of the chunks
chunks = len(ImageList) // size
chunk_size, remainder = divmod(len(ImageList), chunks)
print("chunk_size: " + str(chunk_size) + " remainder: " + str(remainder))
# if remainder>0:
# chunk_size = chunk_size + 1
# remainder = len(ImageList) - (chunk_size * (chunks -1))



print("total images:" + str(len(ImageList)) + ", " + str(chunk_size) + " chunks")

# get start and end indices of the hdf5

# does this over write one space??
if rank == (chunks - 1):
    start = int(rank * chunk_size)
    # end = int(start + remainder)
    end = int(start + remainder + chunk_size)
    print("rank: " + str(rank) + " start: " + str(start) + " end: " + str(end))
else:
    start = int(rank * chunk_size)
    end = int(start + chunk_size)
    print("rank: " + str(rank) + " start: " + str(start) + " end: " + str(end))

# load in data in chunks within this process
ranges = end - start
print("ranges " + str(ranges))
sub_chunk_size, sub_chunk_remainder = divmod(ranges, sub_chunks)
sub_chunk_size = sub_chunk_size + 1
sub_chunk_remainder = ranges - (sub_chunk_size * (sub_chunks -1))


for j in range(0,sub_chunks):
    # get start and end indices of the hdf5
    if j == (sub_chunks - 1):
        sub_start = int(start  + sub_chunk_size * j)
        # sub_end = int(sub_start + sub_chunk_remainder)
        sub_end = int(sub_start + sub_chunk_remainder + sub_chunk_size)
        print("rank: " + str(rank) + " chunk: " + str(j) + " start: " + str(sub_start) + " end: " + str(sub_end))
    else:
        sub_start = int(start + sub_chunk_size * j)
        sub_end = int(sub_start + sub_chunk_size)
        print("rank: " + str(rank) + " chunk: " + str(j) + " start: " + str(sub_start) + " end: " + str(sub_end))

    loadedImages = []
    loadedPatterns = []
    loadedSlides = []
    loadedTiles = []
    loadedLabels = []
    loadedSamples = []
    for i, img in enumerate(ImageList[sub_start:sub_end]):

        image = cv2.imread(img)
        pattern = img.split("/")[-2]
        slide = img.split("/")[-3]
        tile = img.split("/")[-1]
        sample = img.split("/")[-3]
        print("img: " + img + " pattern: " + pattern + " slide: " + slide + " tile: " + tile + " sample: " + sample)


        #    if max(image.shape[0]/float(WIDTH), image.shape[1]/float(HEIGHT)) > 1:  
        #        nfac = 1./max(image.shape[0]/float(WIDTH), image.shape[1]/float(HEIGHT))
        #        try:
        #             image = cv2.resize(image, (0,0), fx=nfac, fy=nfac)
        #        except:
        #             print("error 2:" + img + " " + str(nfac) + " " + str(image.shape[0]) + " " + str(image.shape[1]))
        #    if image.shape[0] > WIDTH:
        #        image = image[:WIDTH,:,:]
        #    if image.shape[1] > HEIGHT:
        #        image =image[:,:HEIGHT,:]
        #    image = np.uint8(image)
        #    sample = slide[:args.sampleID]
        #    slide = slide[:args.slideID]
        #    if image.shape[0] == WIDTH and image.shape[1] == HEIGHT:
        #        loadedImages.append(image)
        #        loadedPatterns.append(pattern)
        #        loadedSlides.append(slide)
        #        loadedTiles.append(tile)
        #        loadedSamples.append(sample)
        #        try:
        #           loadedLabels.append(patterns1.index(pattern))
        #        except:
        #           loadedLabels.append(-1)
        #    else:
            #print("ELSE")
        #    delta_h = HEIGHT - image.shape[0]
        #    delta_w = WIDTH - image.shape[1]
        #    top, remainder_vert = divmod(delta_h, 2)
        #    top = top + remainder_vert
        #    bottom, remainder_vert = divmod(delta_h, 2)
        #    left, remainder_horz = divmod(delta_w, 2)
        #    left = left + remainder_horz
        #    right, remainder_horz = divmod(delta_w, 2)
        #    color = [255, 255, 255]
            #print("cv2:" + str(top) + "_" + str(botton) + "_" + str(right))
        #    #print(cv2.BORDER_CONSTANT)
        #    image = cv2.copyMakeBorder(src=image, top=top,
        #                               bottom=bottom,
        #                               left=left,
        #                               right=right, borderType=cv2.BORDER_CONSTANT, value=color)
        loadedImages.append(image)
        loadedPatterns.append(pattern)
        loadedSlides.append(slide)
        loadedTiles.append(tile)
        #    loadedLabels.append(patterns1.index(pattern))
        loadedSamples.append(sample)

#    print("***********" + str(rank) + "; " + str(sub_chunks) + "; "  + str(j) + "; "  + str(sub_chunk_size) + " to " + str(len(loadedImages)))
   # print(sub_chunks)
   # print(j)
   # print(len(loadedImages))
   # print(len(ImageList[sub_start:sub_end]))
   # print(loadedImages)
        image_stack = np.stack(loadedImages, axis=0)
        pattern_stack = np.stack(loadedPatterns, axis=0).astype('S37')
        slide_stack = np.stack(loadedSlides, axis=0).astype('S23')
        #tile_stack =  np.stack(loadedTiles, axis=0).astype('S9')
        tile_stack =  np.stack(loadedTiles, axis=0).astype('S16')
        # labels_stack = np.stack(loadedLabels, axis=0)
        sample_stack = np.stack(loadedSamples, axis=0).astype('S23')


        dset[sub_start:sub_end, ...] = image_stack
        dset2[sub_start:sub_end, ...] = pattern_stack
        dset3[sub_start:sub_end, ...] = slide_stack
        dset4[sub_start:sub_end, ...] = tile_stack
        # dset5[sub_start:sub_end, ...] = labels_stack
        dset6[sub_start:sub_end, ...] = pattern_stack
        dset7[sub_start:sub_end, ...] = sample_stack

# print("closing:")
# len(dset)
# len(dset2)
# len(dset3)
# len(dset4)
# len(dset5)
    f.close()
    print("job " + str(j) + " finished properly")