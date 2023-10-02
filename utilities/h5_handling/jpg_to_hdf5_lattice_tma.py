

import h5py
import glob
import os
import numpy as np
import cv2
from mpi4py import MPI
from itertools import chain
import argparse


parser = argparse.ArgumentParser(description="Convert jpeg images sorted is subfolders (1 per class) to hdf5 format.")

parser.add_argument("--input", type=str, default='',
                        help="input path ")
parser.add_argument("--output", type=str, default='',
                        help="path and name of output")
parser.add_argument("--wSize", type=int, default=224, help="width and height of images")

parser.add_argument("--subset", type=str, default='validation', help="subset of data")

parser.add_argument("--shuffle", type=bool, default=True, help="shuffle images")

parser.add_argument("--name", type=str, default='lattice_tma', help="name of dataset")


def set_param(args):
    global output_path, input_path, WIDTH, HEIGHT, SHUFFLE, comm, rank, size
    output_path = args.output
    input_path = args.input
    WIDTH = args.wSize
    HEIGHT = args.wSize
    SHUFFLE = args.shuffle

        
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


if __name__ == '__main__':
    args = parser.parse_args()
    set_param(args)
    images = glob.glob(input_path + '/*')
    f = h5py.File(output_path + '/hdf5_{}_he_{}.h5'.format(args.name , args.subset), 'w')
    # Create dataset
    dset = f.create_dataset('img', (len(images), WIDTH, HEIGHT, 3), dtype='uint8')
    dset2 = f.create_dataset('patterns', (len(images), ),
                            dtype = 'S37')
    dset3 = f.create_dataset('slides', (len(images), ),
                            dtype = 'S23')
    dset4 = f.create_dataset('tiles', (len(images), ),
                            dtype = 'S16')
    dset5 = f.create_dataset('labels', (len(images), ))
    dset6 = f.create_dataset('hist_subtype', (len(images), ),
                            dtype = 'S37')
    dset7 = f.create_dataset('samples', (len(images), ),
                         dtype = 'S23')
    
    if SHUFFLE:
        np.random.shuffle(images)
    for i, image in enumerate(images):
        img = cv2.imread(image)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        image = image.split('/')[-1]
        # Name example: MESO_TMA_2_H&E_A-2_1-1.jpg (please change this if you have different names)
        dset[i] = img
        dset2[i] = image.split('_')[0]+'_'+image.split('_')[1]
        dset3[i] = image.split('_')[0] + '_' + image.split('_')[1] + '_' + image.split('_')[2] + '_' + image.split('_')[3] + '_' + image.split('_')[4]
        dset4[i] = image.split('_')[5]
        dset5[i] = 0.0 if args.subset == 'train' else 1.0 if args.subset == 'test' else 2.0
        dset6[i] = image.split('_')[0]+'_'+image.split('_')[1]
        dset7[i] = image.split('_')[4]
    
    for key in f.keys():
        print('--------------------------------------------------')
        print(key)
        print(f[key].shape)
        print(f[key][-1])
    f.close()








