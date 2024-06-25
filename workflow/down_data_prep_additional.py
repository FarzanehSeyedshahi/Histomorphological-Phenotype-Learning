# warning ignore
import warnings
warnings.filterwarnings("ignore")
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import pandas as pd
import random
import shutil
import pandas as pd
import umap
import plotly.express as px
import os
import numpy as np
from tqdm import tqdm
import argparse

# Own libraries
import sys
main_path = '/mnt/cephfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning'
sys.path.append(main_path)
from data_manipulation.data import Data
from data_manipulation.utils import store_data
from models.evaluation.folds import load_existing_split
from models.clustering.data_processing import *

# Add arguments
parser = argparse.ArgumentParser(description='This script takes the clustering output and generate the data for MIL training by names of methods')
parser.add_argument('--dataset', type=str, default='Meso', help='Dataset name')
parser.add_argument('--additional_dataset', default=None , type=str , help='additional_dataset name')

parser.add_argument('--resolutions', type=list, default=[2.0, 0.4, 0.7, 1.0, 1.5, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0], help='Resolutions to use')
parser.add_argument('--meta_folder', type=str, default='subtype', help='meta data folder')
# capitalization of meso_type
parser.add_argument('--meta_field', type=str, default='Meso_type', help='meta data specific field')
parser.add_argument('--matching_field', type=str, default='slides', help='matching field')

# parser.add_argument('h5_additional_path', type=str, help='Path to the additional h5 file')
# parser.add_argument('h5_complete_path', type=str, help='Path to the complete h5 file')

parser.add_argument('--transformation', type=str, default='clr', help='Transformation of the WSI data')
parser.add_argument('--top_variance_feat', type=int, default=99, help='Number of top variance features to keep')
parser.add_argument('--use_conn', type=bool, default=False, help='Use connected components')
parser.add_argument('--use_ratio', type=bool, default=False, help='Use ratio of connected components')
parser.add_argument('--min_tiles', type=int, default=1, help='Minimum number of tiles to keep')
parser.add_argument('--additional_as_fold', type=bool, default=True, help='Use additional data as fold')
parser.add_argument('--force_fold', type=int, default=None, help='Force fold number')
parser.add_argument('--pkl_path', type=str, default=None, help='Path to the pkl file')


if __name__ =='__main__':
    args = parser.parse_args()
    folds_pickle  = args.pkl_path
    meta_field     = args.meta_field
    rep_key        = 'z_latent'
    num_of_folds   = 5
    matching_field = args.matching_field
    resolutions     = args.resolutions
    adatas_path_name = args.meta_folder
    dataset = args.dataset
    additional_dataset = args.additional_dataset

    
    h5_complete_path = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/hdf5_{}_he_complete_metadata.h5'.format(main_path, dataset, dataset)
    h5_additional_path = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/hdf5_{}_he_complete_metadata.h5'.format(main_path, additional_dataset, additional_dataset)
    adatas_path = h5_complete_path.split('/hdf5_')[0] + '/{}/adatas/'.format(adatas_path_name)
    subtype_csvs_path = h5_complete_path.split('/hdf5_')[0] + '/{}/subtype_csvs'.format(adatas_path_name)
    # make if not exist
    if not os.path.exists(subtype_csvs_path): os.makedirs(subtype_csvs_path)
    
    if not os.path.exists(subtype_csvs_path+'/HPC_frames/'): os.makedirs(subtype_csvs_path+'/HPC_frames/')
    
    folds = load_existing_split(folds_pickle)
    for resolution in resolutions:
        print('Resolution: {}'.format(resolution))
        for fold_number in range(num_of_folds):
            groupby        = 'leiden_%s' % resolution
            adata_name         = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), fold_number)
            adata_file_path         = adatas_path + adata_name + '_subsample.h5ad'

            dataframes, complete_df, leiden_clusters = read_csvs(adatas_path, matching_field, groupby, fold_number, folds[fold_number], h5_complete_path, h5_additional_path, force_fold=fold_number)

            # 0 for Epithelioid, 1 for Biphasic or Sarcomatoid
            train_df, valid_df, test_df, additional_df = dataframes
            if additional_dataset == 'MesoGraph':
                # Capitalize the meso_type
                additional_df = additional_df.replace({'Meso_type': {'E': 0, 'B': 1, 'S': 1}})
                print(additional_df)
            
            dataframes = [train_df, None, test_df, additional_df]
            print(dataframes)
            
            type_composition = args.transformation
            if type_composition in ['clr', 'ilr','alr', 'percentage', 'count']:
                top_variance_feat = args.top_variance_feat
                use_conn = args.use_conn
                use_ratio = args.use_ratio
                min_tiles = args.min_tiles
                additional_as_fold = args.additional_as_fold
                force_fold = args.force_fold

                # Create representations per sample: cluster % of total sample.
                # data, data_df, features = prepare_data_classes(dataframes, matching_field, meta_field, groupby, leiden_clusters, type_composition, min_tiles, use_conn=use_conn, use_ratio=use_ratio, top_variance_feat=top_variance_feat)
                # Check clusters and diversity within.
                frame_clusters, frame_samples = create_frames(additional_df, groupby, meta_field, diversity_key=matching_field, reduction=2)
                # Include features that are not the regular leiden clusters.
                

                additional, additional_slides_df, _, features = prepare_set_classes(additional_df, matching_field, meta_field, groupby, leiden_clusters, type_=type_composition, min_tiles=1, use_conn=use_conn, use_ratio=use_ratio, top_variance_feat=top_variance_feat, keep_features=None, return_tiles=False)


                frame_clusters = include_features_frame_clusters(frame_clusters, leiden_clusters, features, groupby)
                frame_clusters.to_csv('{}/HPC_frames/{}_{}_{}_fold{}_hpc_purity.csv'.format(subtype_csvs_path, dataset, type_composition, groupby.replace('.', 'p'), fold_number))




                print('Additional DF: ', additional_df)
                additional_df_aggregated = additional_df.groupby(matching_field).aggregate(lambda x: list(set(x))[0] if len(set(x))==1 else list(x)).reset_index()
                print('Additional DF Aggregated: ', additional_df_aggregated)

                # temp3 = data_df[3].merge(additional_df_aggregated, on=matching_field, how='inner')
                temp3 = additional_slides_df.merge(additional_df_aggregated, on=matching_field, how='inner')
                temp3.to_csv('{}/{}_{}_{}_fold{}.csv'.format(subtype_csvs_path, additional_dataset, type_composition, groupby.replace('.', 'p'), fold_number))