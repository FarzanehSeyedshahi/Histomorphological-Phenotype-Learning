

import h5py
import glob
import os
import numpy as np
import cv2
from mpi4py import MPI
from itertools import chain
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Convert jpeg images sorted is subfolders (1 per class) to hdf5 format.")

parser.add_argument("--input", type=str, default='',
                        help="input path ")
parser.add_argument("--output", type=str, default='',
                        help="path and name of output")
parser.add_argument("--wSize", type=int, default=224, help="width and height of images")

parser.add_argument("--shuffle", type=bool, default=True, help="shuffle images")

parser.add_argument("--name", type=str, default='TCGA_MESO', help="name of dataset")
parser.add_argument("--subset", type=str, default='combined', help="subset of data")


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
    combined_imgs = glob.glob(input_path + '/*')
    if SHUFFLE:
        np.random.shuffle(combined_imgs)

    test_imgs, train_imgs, valid_imgs = [], [], []
    for i, image in tqdm(enumerate(combined_imgs)):
        image_type = image.split('/')[-1].split('_')[0]
        if image_type == 'test':
            test_imgs.append(image)
        elif image_type == 'train':
            train_imgs.append(image)
        elif image_type == 'valid':
            valid_imgs.append(image)

    print('test: ', len(test_imgs))
    print('train: ', len(train_imgs))
    print('valid: ', len(valid_imgs))
    print('combined: ', len(combined_imgs))

    subset_list = ['test', 'train', 'validation', 'combined']
    for subset, images in tqdm(zip(subset_list, [test_imgs, train_imgs, valid_imgs, combined_imgs])):
        # Create hdf5 file
        f = h5py.File(output_path + '/hdf5_{}_he_{}.h5'.format(args.name , subset), 'w')
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
        


        for i, image in tqdm(enumerate(images)):
            img = cv2.imread(image)
            img = cv2.resize(img, (WIDTH, HEIGHT))
            image = image.split('/')[-1]
            # new name example: valid_TCGA-ZN-A9VO-01Z-00-DX1.6CE41901-D74E-404F-89D3-7DDB97378854_9_43.jpeg
            dset[i] = img
            dset2[i] = image.split('_')[1].split('.')[1]
            dset3[i] = image.split('_')[1].split('.')[0]
            dset4[i] = image.split('_')[2] + '-' + image.split('_')[3]
            # test:0.0, train:1.0, validation:2.0
            dset5[i] = 0.0 if image.split('_')[0] == 'train' else 1.0 if image.split('_')[0] == 'test' else 2.0
            dset6[i] = image.split('_')[0]
            temp = image.split('_')[1].split('.')[0].split('-')
            dset7[i] = temp[0] + '-' + temp[1] + '-' + temp[2]

        for key in f.keys():
            print('--------------------------------------------------')
            print(key)
            print(f[key].shape)
            print(f[key][-1])
            print(f[key][0])
        f.close()

    