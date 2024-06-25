from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import csv
import os
import gc
from tqdm import tqdm

# Own libs
import sys
main_path = '/mnt/cephfs/sharedscratch/users/fshahi/Projects/Histomorphological-Phenotype-Learning'
sys.path.append(main_path)
from data_manipulation.data import Data
from models.clustering.data_processing import *
from models.visualization.attention_maps import *
from models.clustering.leiden_representations import include_tile_connections_frame






def cluster_set_images(frame, data_dicts, groupby, set_, leiden_path, dpi):
    images_path    = os.path.join(leiden_path, 'images')
    backtrack_path = os.path.join(leiden_path, 'backtrack')
    if not os.path.isdir(images_path):
        os.makedirs(images_path)
        os.makedirs(backtrack_path)

    for cluster_id in pd.unique(frame[groupby].values):
        indexes       = frame[frame[groupby]==cluster_id]['indexes'].values.tolist()
        original_sets = frame[frame[groupby]==cluster_id]['original_set'].values.tolist()
        combined      = list(zip(indexes, original_sets))
        random.shuffle(combined)

        csv_information = list()
        images_cluster = list()
        i = 0
        # print('Data dict keys: %s' % data_dicts)

        for index, original_set in tqdm(combined):
            # print('Index: %s, Original set: %s' % (index, original_set))
            images_cluster.append(data_dicts[original_set][int(index)]/255.)
            csv_information.append(frame[(frame.indexes==index)&(frame.original_set==original_set)].to_dict('index'))
            i += 1
            if i==100:
                break

        sns.set_theme(style='white')
        fig = plt.figure(figsize=(40, 8))
        fig.suptitle('Cluster %s' % (cluster_id), fontsize=18, fontweight='bold')
        grid = ImageGrid(fig, 111, nrows_ncols=(5, 20), axes_pad=0.1,)

        for ax, im in zip(grid, images_cluster):
            ax.imshow(im)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticks([])

        plt.savefig(os.path.join(images_path, 'cluster_%s_%s.jpg' % (cluster_id, set_)), dpi=dpi)
        plt.close(fig)
        sns.set_theme(style='darkgrid')

        # Tracking file for selected images.
        with open(os.path.join(backtrack_path, 'set_%s_%s.csv' % (cluster_id, set_)), 'w') as content:
            w = csv.DictWriter(content, frame.columns.to_list())
            w.writeheader()
            for element in csv_information:
                for index in element:
                    w.writerow(element[index])



def plot_cluster_images(groupby, meta_folder, data, fold, h5_complete_path, dpi, value_cluster_ids, extensive=True):
    main_cluster_path = h5_complete_path.split('hdf5_')[0]
    main_cluster_path = os.path.join(main_cluster_path, meta_folder)
    adatas_path       = os.path.join(main_cluster_path, 'adatas')
    leiden_path       = os.path.join(main_cluster_path, '%s' % (groupby.replace('.', 'p')))
    if not os.path.isdir(leiden_path):
        os.makedirs(leiden_path)
        
    # Data Class with all h5, these contain the images.
    data_dicts = dict()
    data_dicts['train'] = data.training.images
    data_dicts['valid'] = None
    if data.validation is not None:
        data_dicts['valid'] = data.validation.images
    data_dicts['test'] = None
    if data.test is not None:
        data_dicts['test'] = data.test.images

    # Base name for each set representations.
    adata_name     = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s' % (groupby.replace('.', 'p'))

    # Train set.
    train_fold_csv = os.path.join(adatas_path, '%s_train.csv' % adata_name)
    if not os.path.isfile(train_fold_csv):
        train_fold_csv = os.path.join(adatas_path, '{}__fold{}.csv'.format(adata_name, fold))
    print('Train fold csv: %s' % train_fold_csv)
    frame_train    = pd.read_csv(train_fold_csv)
    # Annotations csv.
    # create_annotation_file(leiden_path, frame_train, groupby, value_cluster_ids)
    cluster_set_images(frame_train, data_dicts, groupby, 'train', leiden_path, dpi)

    # Test set.
    # if extensive:
    #     test_fold_csv = os.path.join(adatas_path, '%s_test.csv' % adata_name)
    #     frame_test    = pd.read_csv(test_fold_csv)
    #     cluster_set_images(frame_test, data_dicts, groupby, 'test', leiden_path, dpi)


dbs_path = '/mnt/cephfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning'
dataset = 'Meso_500'
refrence_dataset = 'Meso_400_subsampled'
model_name = 'BarlowTwins_3'
h5_complete_path = '{}/results/{}/{}/h224_w224_n3_zdim128/hdf5_{}_he_complete_metadata_filtered.h5'.format(dbs_path, model_name, refrence_dataset, dataset)
meta_folder       = 'meso_subtypes_nn400'
rep_key          = 'z_latent'
resolution      = 2.0
# Leiden convention name.
groupby = 'leiden_%s' % resolution

# Dataset images.
marker  = 'he'
image_width  = 224
image_height = 224
image_channels = 3
data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=64, project_path=dbs_path, load=True)
print('Loaded dataset: %s' % data.training)
print('Loaded dataset: %s' % data)
value_cluster_ids = dict()
# Dump cluster images.
plot_cluster_images(groupby, meta_folder, data, fold=0, h5_complete_path=h5_complete_path, dpi=1000, value_cluster_ids=value_cluster_ids, extensive=None)




