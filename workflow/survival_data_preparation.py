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
parser.add_argument('--dataset', type=str, default='Meso_500', help='Dataset name')
parser.add_argument('--additional_dataset', default=None, type=str , help='additional_dataset name')

parser.add_argument('--resolutions', type=list, default=[2.0, 0.4, 0.7, 1.0, 1.5, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0], help='Resolutions to use')
parser.add_argument('--meta_folder', type=str, default='survival', help='meta data folder')
parser.add_argument('--meta_field', type=str, default='os_event_ind', help='meta data specific field')
parser.add_argument('--matching_field', type=str, default='case_Id', help='matching field')
parser.add_argument('--event_ind_field', type=str, default='os_event_ind', help='event indicator field')
parser.add_argument('--event_data_field', type=str, default='os_event_data', help='event data field')

# parser.add_argument('h5_additional_path', default=None, type=str, help='Path to the additional h5 file')
# parser.add_argument('h5_complete_path', type=str, help='Path to the complete h5 file')

parser.add_argument('--transformation', type=str, default='clr', help='Transformation of the WSI data')
parser.add_argument('--top_variance_feat', type=int, default=99, help='Number of top variance features to keep')
parser.add_argument('--use_conn', type=bool, default=False, help='Use connected components')
parser.add_argument('--use_ratio', type=bool, default=False, help='Use ratio of connected components')
parser.add_argument('--min_tiles', type=int, default=10, help='Minimum number of tiles to keep')
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
    event_ind_field    = args.event_ind_field
    event_data_field   = args.event_data_field
    dataset = args.dataset
    additional_dataset = args.additional_dataset
    if additional_dataset is not None:
        h5_additional_path   = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/hdf5_{}_he_complete_filtered_metadata.h5'.format(main_path, additional_dataset, additional_dataset)
    else:
        h5_additional_path = None
    
    # h5_complete_path = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128_filtered/hdf5_{}_he_complete_filtered_metadata.h5'.format(main_path, dataset, dataset)
    h5_complete_path = './results/BarlowTwins_3/Meso/h224_w224_n3_zdim128/hdf5_Meso_he_complete_filtered_metadata.h5'

    adatas_path = h5_complete_path.split('/hdf5_')[0] + '/{}/adatas/'.format(adatas_path_name)
    survival_csvs_path = h5_complete_path.split('/hdf5_')[0] + '/{}/survival_csvs'.format(adatas_path_name)
    if not os.path.exists(survival_csvs_path):
        os.makedirs(survival_csvs_path)
    if not os.path.exists('{}/HPC_frames'.format(survival_csvs_path)):
        os.makedirs('{}/HPC_frames'.format(survival_csvs_path))
    
    folds = load_existing_split(folds_pickle)
    for resolution in resolutions:
        print('Resolution: {}'.format(resolution))
        for fold_number in range(num_of_folds):
            groupby        = 'leiden_%s' % resolution
            adata_name         = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), fold_number)
            adata_file_path         = adatas_path + adata_name + '_subsample.h5ad'

            dataframes, complete_df, leiden_clusters = read_csvs(adatas_path, matching_field, groupby, fold_number, folds[fold_number], h5_complete_path, h5_additional_path, force_fold=fold_number)
            complete_df['tiles']   = complete_df['tiles'].apply(lambda x: x.split('.jpeg')[0])
            
            for df in dataframes:
                if df is not None:
                    df.replace("-", np.inf, inplace=True)
                    df.replace("'--", np.inf, inplace=True)
            # making it to months (IN TCGA DATA)
            dataframes[-1][event_data_field]= dataframes[-1][event_data_field].apply(lambda x: round(float(x)/30.0,2) if x != np.inf else x)
            for df in dataframes:
                print(df)
                print('-------------------')

            type_composition = args.transformation
            if type_composition in ['clr', 'ilr','alr', 'percentage']:
                # Use connectivity between clusters as features.
                top_variance_feat = args.top_variance_feat
                use_conn = args.use_conn
                use_ratio = args.use_ratio
                min_tiles = args.min_tiles
                additional_as_fold = args.additional_as_fold
                force_fold = args.force_fold
                
                remove_clusters = None
                # Other features
                q_buckets  = 2
                max_months = 15.0*15.0

                # # Check clusters and diversity within.
                frame_clusters, frame_samples = create_frames(complete_df, groupby, meta_field, diversity_key=matching_field, reduction=2)

                list_df, list_all_df, features = prepare_data_survival(dataframes, groupby, leiden_clusters, type_composition, max_months, matching_field, event_ind_field, event_data_field, min_tiles,
																use_conn=use_conn, use_ratio=use_ratio, top_variance_feat=top_variance_feat, remove_clusters=remove_clusters)

                # Include features that are not the regular leiden clusters.
                # frame_clusters = include_features_frame_clusters(frame_clusters, leiden_clusters, features, groupby)
                frame_clusters.to_csv('{}/HPC_frames/{}_{}_{}_fold{}_hpc_purity.csv'.format(survival_csvs_path, dataset, type_composition, groupby.replace('.', 'p'), fold_number))
                if additional_dataset is not None:
                    frame_clusters.to_csv('{}/HPC_frames/{}_{}_{}_fold{}_hpc_purity.csv'.format(survival_csvs_path, additional_dataset, type_composition, groupby.replace('.', 'p'), fold_number))

                

                # # Groupby matching field and save the all metadata per matching field. [if there is one type, put that type, if not put all types as a list]
                # complete_df_aggregated = complete_df.groupby(matching_field).aggregate(lambda x: list(set(x))[0] if len(set(x))==1 else list(x)).reset_index()
                # if additional_df is not None:
                #     additional_df_aggregated = additional_df.groupby(matching_field).aggregate(lambda x: list(set(x))[0] if len(set(x))==1 else list(x)).reset_index()

                # # add feature vector to the aggregated dataframes
                # temp = pd.concat([data_df[0], data_df[1], data_df[2]])
                # temp2 = temp.merge(complete_df_aggregated, on=matching_field, how='inner')
                # # the difference: array(['MESO_328_5', 'MESO_349_5', 'MESO_353_4(2)'], dtype='<U16')
                # temp2.to_csv('{}/{}_{}_{}_fold{}.csv'.format(survival_csvs_path, dataset, type_composition, groupby.replace('.', 'p'), fold_number))


                # Add metadata to the dataframes.
                adding_metadata = True
                if adding_metadata:
                    # metadata_list = ['case_Id', 'HB_score', 'Meso_type', 'Sex', 'TNM_Stage', 'N_stage', 'T_Stage', 'chest_wall_involvement', 'time_to_recurrence', 'wcc_score', 'smoking_history' ]
                    metadata_list = ['case_Id', 'Stage', 'Sex', 'age', 'Meso_type']
                    for df in list_all_df:
                        print(df[1])
                        print(df[0])
                        if df[1] == 'additional':
                            csv_patient = pd.read_csv('./files/TCGA_files/clinical_TCGA_clean.csv')
                        else:
                            csv_patient = pd.read_csv('./files/Meso_patients.csv')
                        if df[0] is not None:
                            tmp = df[0].merge(csv_patient[metadata_list], on=matching_field, how='inner')
                            tmp.to_csv('{}/{}_{}_fold{}_{}_metadata.csv'.format(survival_csvs_path, type_composition, groupby.replace('.', 'p'), fold_number, df[1]))
                            # duration_col=event_data_field, event_col=event_ind_field


                else:
                    for df in list_all_df:
                        if df[0] is not None:
                            df[0].to_csv('{}/{}_{}_fold{}_{}.csv'.format(survival_csvs_path, type_composition, groupby.replace('.', 'p'), fold_number, df[1]))


