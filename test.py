import wandb
import pandas as pd
import numpy as np
import h5py
import pickle
import math
import os
from utilities.fold_creation.fold_cross_validation import *

main_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning'
# # h5_path   = '%s/results/BarlowTwins_3/Meso_400_subsampled/h224_w224_n3_zdim128/hdf5_Meso_400_subsampled_he_complete_combined_metadata_filtered.h5' % main_path
# h5_complete_path = '%s/results/BarlowTwins_3/Meso_400_subsampled/h224_w224_n3_zdim128/hdf5_Meso_400_subsampled_he_complete_filtered_combined_metadata.h5'%main_path

# file_h5_complete = h5py.File(h5_complete_path, 'r')
# for key in file_h5_complete.keys():
#     print('--------------------------------------------------')
#     print(key)
#     print(file_h5_complete[key].shape)
#     print(file_h5_complete[key][-1])



representation_csv='%s/files/csv_Meso_400_subsampled_he_complete_filtered.csv' % main_path
patient_csv='%s/files/Mesothelioma_patients_labels_full.csv' % main_path
csv_results_path = '%s/files/csv_Meso_400_subsampled_he_complete_filtered_with_survival.csv' % main_path

# csv_df = pd.read_csv(csv_results_path, index_col=0)
# for col in csv_df.columns:
#     print(col, csv_df[col].unique())

# save the columns as string
# csv_df['wcc_score'] = csv_df['wcc_score'].astype(str)
# csv_df['wcc_score'] = csv_df['wcc_score'].replace('nan', '0.0')



if not os.path.exists(representation_csv):
    print('Creating csv file from h5 file')
    df_representations = creat_csv_from_h5(representation_csv, h5_complete_path)

# add the survival information to the csv file
os.remove(csv_results_path)
df_representations = create_csv(patient_csv=patient_csv\
                                , representation_csv=representation_csv,\
                                    csv_results_path=csv_results_path)


csv_df = pd.read_csv(csv_results_path, index_col=0)
for col in csv_df.columns:
    print(col, csv_df[col].unique())
# from sklearn.datasets import load_digits

# wandb.init(project="embeddings", entity="farzaneh_uog")

# Load the dataset
# ds = load_digits(as_frame=True)
# df = ds.data

# # Create a "target" column
# df["target"] = ds.target.astype(str)
# cols = df.columns.tolist()
# df = df[cols[-1:] + cols[:-1]]

# # Create an "image" column
# # .loc[row_indexer,col_indexer] = value
# df["image"] = df.apply(lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1)
# cols = df.columns.tolist()
# df = df[cols[-1:] + cols[:-1]]


# hpl_df = pd.read_csv('./datasets/clf_csvs/data_label_res-2.0_fold-0.csv', index_col=0)

# print(hpl_df)
# # rename columns
# hpl_df.rename(columns={'label': 'target'}, inplace=True)
# hpl_df['target'] = hpl_df['target'].astype(str)
# # df = hpl_df.drop(columns=['label'])
# # df["target"] = hpl_df.label.astype(str)
# # cols = df.columns.tolist()
# # df = df[cols[-1:] + cols[:-1]]



# wandb.log({"HPL": hpl_df})
# wandb.finish()