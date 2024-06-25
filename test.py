import wandb
<<<<<<< HEAD
import wandb
=======
>>>>>>> b9ab7308fe30b059bd3fb6a2566c7246f9663a8c
import pandas as pd
import numpy as np
import h5py
import pickle
import math
import os
<<<<<<< HEAD

main_path = '/mnt/cephfs/sharedscratch/users/fshahi/Projects/Histomorphological-Phenotype-Learning'
import matplotlib.pyplot as plt
import numpy as np

def umap_plot(X,y, title='UMAP projection of the dataset'):
    # draw umap from X and color it based on y
    import umap
    import plotly.express as px
    reducer = umap.UMAP(n_components=3)
    print('UMAP embedding...')
    embedding = reducer.fit_transform(X)
    fig = px.scatter(
        embedding, x=0, y=1, color=y,
        title='UMAP projection of the dataset', labels={'color': 'digit'}
    )
    # wandb.log({"UMAP": fig})
    fig.show()
    fig.write_html('umap_{}.html'.format(title))
    print('{}UMAP embedding saved as umap_{}.png'.format(main_path,title))


# dataset = 'Meso'
# # dataset = 'Meso_250_subsampled'
# # dataset = 'TCGA_MESO'
# h5_complete_path   = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/hdf5_{}_he_complete_metadata.h5'.format(main_path, dataset, dataset)

# df1 = pd.read_csv('{}/files/Meso_patients.csv'.format(main_path))
# df2 = pd.read_csv('{}/files/Mesothelioma_patients_labels_full.csv'.format(main_path))
# df_temp = df2[['Age at Surgery', 'Sex', 'Case Number', 'Follow-Up Status', 'Time to Follow-Up Status (Months)']]
# df_temp = df_temp.rename(columns={'Age at Surgery':'age_at_surgery', 'Sex':'gender', 'Case Number':'case_Id', 'Follow-Up Status':'recurrnce', 'Time to Follow-Up Status (Months)': 'time_to_recurrence'})
# df_temp['recurrnce'] = df_temp['recurrnce'].replace({'No Recurrence': 0, 'Recurrence': 1})
# df1.merge(df_temp, on='case_Id', how='left').to_csv('{}/files/Meso_patients.csv'.format(main_path), index=False)

# add 


from utilities.fold_creation.fold_cross_validation import *

h5_complete_path   = '%s/results/BarlowTwins_3/Meso_500/h224_w224_n3_zdim128/hdf5_Meso_500_he_complete.h5' % main_path
# h5_complete_path = '{}/datasets/Meso_500/he/patches_h224_w224/hdf5_Meso_500_he_train.h5'.format(main_path)
representation_csv='%s/files/csv_Meso_500_he_complete.csv' % main_path
patient_csv='%s/files/Mesothelioma_patients_labels_full.csv' % main_path
csv_results_path = '%s/files/csv_Meso_500_he_complete_with_survival.csv' % main_path
Meso_patients_csv = '%s/files/Meso_patients.csv' % main_path


# if not os.path.exists(representation_csv):
    # print('Creating csv file from h5 file')
# df_representations = creat_csv_from_h5_wsi(representation_csv, h5_complete_path)
# else:
    # df_representations = pd.read_csv(representation_csv)

# add the survival information to the csv file
# df_representations = create_csv_wsi(df_representations=df_representations, patient_csv=patient_csv, csv_results_path=csv_results_path)

# # adding case_id based on samples to the hdf5 file
# from tqdm import tqdm
# content = h5py.File(h5_complete_path, 'a')
# samples = content['samples'][:]
# content.create_dataset('case_Id', shape=(samples.shape[0],), dtype='S23')
# for i in tqdm(range(samples.shape[0])):
#     case_id = samples[i].decode("utf-8").split('_')[1]
#     content['case_Id'][i] = case_id
# content.close()



h5_complete_path = './results/BarlowTwins_3/Meso_400_subsampled/h224_w224_n3_zdim128_filtered/hdf5_Meso_500_he_complete_metadata_filtered.h5'
file = h5py.File(h5_complete_path, 'r')
for key in file.keys():
    print(key)
    print(file[key][0])
    print(file[key][-1])
    print(file[key].shape)
    print(file[key].dtype)
    print()
    # if key == 'samples':
    #     final_df = pd.DataFrame({'samples':file[key][:].astype(str)})
    #     x_folatsIndeces = final_df['samples'].apply(lambda x: True if len(x.split('_'))<2 else False)
    #     print(final_df[x_folatsIndeces])
    
# check the number of case_ids in csv
# df_representations = pd.read_csv(csv_results_path)
# meso_patients = pd.read_csv(Meso_patients_csv)
# print(df_representations['case_number'].nunique())
# print(sorted(df_representations['case_number'].unique()))
# l = []
# for i in range(0, 512):
#     if i not in df_representations['case_number'].unique():
#         l.append(i)

# meso_patients = meso_patients[meso_patients['case_Id'].isin(df_representations['case_number'].unique())]
# print(meso_patients)
# meso_patients.to_csv(Meso_patients_csv.replace('.csv','meso_500.csv'), index=False)

# reading pkl file 
# import pickle
# print('Reading pkl file')
# dataset = 'Meso_500'
# with open('{}/files/indexes_to_remove/Meso_400_subsampled/{}.pkl'.format(main_path, dataset), 'rb') as f:
#     data = pickle.load(f)
# print(len(data))
# print(np.max(data))


# ['MESO_122_11', 'MESO_145_16', 'MESO_155_8', 'MESO_221_30', 'MESO_136_15', 'MESO_144_18', 'MESO_104_4', 'MESO_138_13', 'MESO_182_17', 'MESO_145_24', 'MESO_171_3', 'MESO_144_16', 'MESO_157_28', 'MESO_172_14', 'MESO_221_25', 'MESO_143_25', 'MESO_148_28', 'MESO_173_20', 'MESO_142_21', 'MESO_110_1', 'MESO_103_33', 'MESO_111_3', 'MESO_140_16', 'MESO_167_39', 'MESO_149_37', 'MESO_148_25', 'MESO_179_70', 'MESO_142_19', 'MESO_160_8', 'MESP_105_11', 'MESO_102_1', 'MESO_221_21', 'MESO_179_73', 'MESO_110_4', 'MESO_173_12', 'MESO_132_21', 'MESO_162_1', 'MESO_158_14', 'MESO_128_5', 'MESO_113_21', 'MESO_158_9', 'MESO_138_12', 'MESO_162_6', 'MESO_158_16', 'MESO_159_12', 'MESO_120_1', 'MESO_162_7', 'MESO_111_1', 'MESO_150_24', 'MESO_149_26', 'MESO_167_46', 'MESO_158_21', 'MESO_165_4', 'MESO_181_13', 'MESO_181_14', 'MESO_182_24', 'MESO_161_16', 'MESO_111_30', 'MESO_140_15', 'MESO_115_8', 'MESO_171_2', 'MESO_132_24', 'MESO_104_6', 'MESO_101_19', 'MESO_150_27', 'MESO_146_17', 'MESO_139_17', 'MESO_103_36', 'MESO_109_1', 'MESO_145_18', 'MESO_168_10', 'MESO_151_14', 'MESO_221_15', 'MESO_180_7', 'MESO_145_14', 'MESO_138_18', 'MESO_138_16', 'MESO_138_20', 'MESO_148_15', 'MESO_179_68', 'MESO_113_16', 'MESO_128_10', 'MESO_168_15', 'MESO_221_27', 'MESO_142_24', 'MESO_125_25', 'MESO_125_24', 'MESO_103_35', 'MESO_179_69', 'MESO_157_41', 'MESO_169_25', 'MESO_160_5', 'MESO_121_11', 'MESO_179_71', 'MESO_169_50', 'MESO_148_34', 'MESO_166_50', 'MESO_162_8', 'MESO_125_19', 'MESO_114_2', 'MESO_103_30', 'MESO_101_30', 'MESO_145_13', 'MESO_166_29', 'MESO_101_18', 'MESO_149_30', 'MESO_183_19', 'MESO_169_5', 'MESO_111_2', 'MESO_138_14', 'MESO_138_10', 'MESO_113_24', 'MESO_143_2', 'MESO_132_20', 'MESO_154_29', 'MESO_160_2', 'MESO_221_31', 'MESO_136_14', 'MESO_122_17', 'MESO_106_3', 'MESO_120_6', 'MESO_120_5', 'MESO_156_7', 'MESO_173_18', 'MESO_103_29', 'MESO_101_21', 'MESO_126_38', 'MESO_138_15', 'MESO_102_3', 'MESO_128_6', 'MESO_128_12', 'MESO_119_8', 'MESO_148_29', 'MESO_147_3', 'MESO_169_6', 'MESO_143_1', 'MESO_159_17', 'MESO_147_1', 'MESO_150_19', 'MESO_122_19', 'MESO_114_4', 'MESO_180_8', 'MESO_113_13', 'MESO_145_20', 'MESO_159_23', 'MESO_103_26', 'MESO_180_17', 'MESO_145_17', 'MESO_146_18', 'MESO_181_11', 'MESO_125_17', 'MESO_221_26', 'MESO_146_12', 'MESO_140_11', 'MESO_149_25', 'MESO_157_36', 'MESO_173_16', 'MESO_175_6', 'MESO_157_27', 'MESO_177_4', 'MESO_158_19', 'MESO_179_72', 'MESO_161_23', 'MESO_162_2', 'MESO_185_26', 'MESO_120_3', 'MESO_167_51', 'MESO_101_16', 'MESO_175_4', 'MESO_158_13', 'MESO_128_11', 'MESO_180_16', 'MESO_165_9', 'MESO_123_1', 'MESO_172_15', 'MESO_118_12', 'MESO_103_25', 'MESO_101_25', 'MESO_156_2', 'MESO_142_17', 'MESO_150_21', 'MESO_157_40', 'MESO_141_12', 'MESO_101_23', 'MESO_103_24', 'MESO_121_17', 'MESO_168_11', 'MESO_113_29', 'MESO_145_22', 'MESO_146_11', 'MESO_111_4', 'MESO_150_26', 'MESO_124_27', 'MESO_120_8', 'MESO_103_23', 'MESO_106_2', 'MESO_145_21', 'MESO_148_14', 'MESO_113_18', 'MESO_105_12', 'MESO_144_12', 'MESO_127_26', 'MESO_168_3', 'MESO_150_30', 'MESO_155_2', 'MESO_138_19', 'MESO_127_17', 'MESO_134_19', 'MESO_113_28', 'MESO_151_22', 'MESO_173_19', 'MESO_118_15', 'MESO_104_1', 'MESO_189_13', 'MESO_159_11', 'MESO_173_15', 'MESO_171_4', 'MESO_124_19', 'MESO_145_9', 'MESO_155_10', 'MESO_121_15', 'MESO_148_35', 'MESO_158_11', 'MESO_101_17', 'MESO_140_23', 'MESO_125_18', 'MESO_151_18', 'MESO_159_15', 'MESO_150_25', 'MESO_186_3', 'MESO_167_50', 'MESO_128_4', 'MESO_159_19', 'MESO_160_6', 'MESO_128_9', 'MESO_121_13', 'MESO_124_25', 'MESO_112_5', 'MESO_124_18', 'MESO_124_21', 'MESO_103_32', 'MESO_126_43', 'MESO_150_20', 'MESO_172_17', 'MESO_150_18', 'MESO_181_9', 'MESO_132_25', 'MESO_166_26', 'MESO_149_28', 'MESO_112_2', 'MESO1_135_17', 'MESO_144_17', 'MESO_124_20', 'MESO_179_66', 'MESO_103_28', 'MESO_158_20', 'MESO_159_14', 'MESO_172_18', 'MESO_158_21 (2)', 'MESO_149_38', 'MESO_107_3', 'MESO_142_13', 'MESO_181_10', 'MESO_101_26', 'MESO_160_1', 'MESO_162_9', 'MESO_101_24', 'MESO_135_3', 'MESO_113_14', 'MESO_125_26', 'MESO_172_13', 'MESO_185_25', 'MESO_136_7', 'MESO_127_16', 'MESO_135_18', 'MESO_134_23', 'MESO_151_19', 'MESO_112_3', 'MESO_183_17', 'MESO_157_35', 'MESO_149_33', 'MESO_145_23', 'MESO_118_20', 'MESO_113_12', 'MESO_114_5', 'MESO_181_8', 'MESO_221_29', 'MESO_143_19', 'MESO_126_35', 'MESO_123_2', 'MESO_173_21', 'MESO_146_14', 'MESO_133_8(2)', 'MESO_103_27', 'MESO_111_11', 'MESO_172_20', 'MESO_128_13', 'MESO_169_10', 'MESO_152_26', 'MESO_130_6', 'MESO_172_23', 'MESO_114_3', 'MESO_176_5', 'MESO_159_18', 'MESO_150_22', 'MESO_143_3', 'MESO_126_40', 'MESO_150_29', 'MESO_112_4', 'MESO_221_33', 'MESO_109_3', 'MESO_148_17', 'MESO_153_31', 'MESO_181_15', 'MESO_132_23', 'MESO_181_18', 'MESO_111_31', 'MESO_173_13', 'MESO_158_18', 'MESO_104_2', 'MESO_168_14', 'MESO_159_22', 'MESO_186_5', 'MESO_120_9', 'MESO_158_17', 'MESO_180_15', 'MESO_118_24', 'MESO_131_2', 'MESO_165_11', 'MESO_118_27', 'MESO_159_13', 'MESO_144_10', 'MESO_135_2', 'MESO_143_20', 'MESO_181_22', 'MESO_167_44', 'MESO_180_14', 'MESO_110_5', 'MESO_104_3', 'MESO_144_11', 'MESO_158_8', 'MESO_114_1', 'MESO_178_13', 'MESO_102_5', 'MESO_179_67', 'MESO_143_18', 'MESO_140_10', 'MESO_143_16', 'MESO_128_7', 'MESO_146_16', 'MESO_146_13', 'MESO_144_13', 'MESO_140_20', 'MESO_183_18', 'MESO_139_14', 'MESO_145_11', 'MESO_140_13', 'MESO_169_49', 'MESO_143_17', 'MESO_164_14', 'MESO_160_9', 'MESO_160_3', 'MESO_102_4', 'MESO_185_31', 'MESO_101_22', 'MESO_173_17', 'MESO_113_27', 'MESO_162_4', 'MESO_138_11', 'MESO_221_24', 'MESO_157_37', 'MESO_149_36', 'MESO_148_27', 'MESO_148_16', 'MESO_135_4', 'MESO_114_7', 'MESO_179_64', 'MESO_126_39', 'MESO_154_25', 'MESO_108_12', 'MESO_181_12', 'MESO_156_6', 'MESO_154_15', 'MESO_167_32', 'MESO_121_14', 'MESO_141_22', 'MESO_173_11', 'MESO_128_8', 'MESO_104_5', 'MESO_101_20', 'MESO_112_1', 'MESO_178_26', 'MESO_101_31', 'MESO_172_16', 'MESO_172_22', 'MESO_172_21', 'MESO_109_2', 'MESO_146_19', 'MESO_157_29', 'MESO_221_19', 'MESO_135_7', 'MESO_120_4', 'MESO_103_34', 'MESO_140_14', 'MESO_102_2', 'MESO_135_1', 'MESO_127_21', 'MESO_184_5', 'MESO_127_27', 'MESO_121_19', 'MESO_173_14', 'MESO_107_4', 'MESO_146_15', 'MESO_101_27', 'MESO_142_23', 'MESO_142_15', 'MESO_161_24', 'MESO_160_10', 'MESO_148_22', 'MESO_113_22', 'MESO_134_16', 'MESO_151_24', 'MESO_127_24', 'MESO_148_24', 'MESO_146_20', 'MESO_134_22', 'MESO_144_15', 'MESO_182_16', 'MESO_221_28', 'MESO_143_12', 'MESO_160_11', 'MESO_126_30', 'MESO_172_19', 'MESO_146_24', 'MESO_178_29', 'MESO-158_12', 'MESO_154_6', 'MESO_179_65']
=======
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
>>>>>>> b9ab7308fe30b059bd3fb6a2566c7246f9663a8c
