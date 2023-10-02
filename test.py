import wandb
import pandas as pd
import numpy as np
import h5py
import pickle
import math
import os

main_path = '/mnt/cephfs/sharedscratch/users/fshahi/Projects/'

import matplotlib.pyplot as plt

import numpy as np


# now read a text file and get the file names column to compare with the list of slides
# text_path = '%sHistomorphological-Phenotype-Learning/files/TCGA_files/clinical_TCGA.tsv' % main_path
# df = pd.read_csv(text_path, sep='\t')
# df = df[['case_id', 'case_submitter_id', 'gender', 'age_at_index', 'days_to_birth', 'days_to_death', 'ethnicity', 'race', 'vital_status', 'year_of_birth', 'year_of_death', 'days_to_last_follow_up',\
#           'age_at_diagnosis', 'ajcc_pathologic_m', 'ajcc_pathologic_n',  'ajcc_staging_system_edition', 'morphology', 'primary_diagnosis', 'prior_malignancy', 'prior_treatment', 'year_of_diagnosis'  ]]


# df = df.groupby(['case_id']).first().reset_index()
# df['primary_diagnosis'] = df['primary_diagnosis'].apply(lambda x: 'Epithelioid' if x=='Epithelioid mesothelioma, malignant' else x)
# df['primary_diagnosis'] = df['primary_diagnosis'].apply(lambda x: 'Biphasic' if x=='Mesothelioma, biphasic, malignant' else x)
# df['primary_diagnosis'] = df['primary_diagnosis'].apply(lambda x: 'Sarcomatoid' if x=='Fibrous mesothelioma, malignant' else x)
# # set one case by one case
# df.loc[df['case_submitter_id']=='TCGA-LK-A4O0', 'primary_diagnosis'] = 'Sarcomatoid'
# df.loc[df['case_submitter_id']=='TCGA-LK-A4NW', 'primary_diagnosis'] = 'Epithelioid'
# df.loc[df['case_submitter_id']=='TCGA-UD-AAC4', 'primary_diagnosis'] = 'Epithelioid'
# df.loc[df['case_submitter_id']=='TCGA-YS-AA4M', 'primary_diagnosis'] = 'Epithelioid'
# df.loc[df['case_submitter_id']=='TCGA-NQ-A638', 'primary_diagnosis'] = 'Epithelioid'
# df = df.rename(columns={'case_id':'patterns', 'days_to_death':'os_event_data', 'primary_diagnosis':'type', 'ajcc_pathologic_n': 'stage' })
# df['os_event_ind'] = df['vital_status'].apply(lambda x: 1 if  x=='Dead' else 0 if x=='Alive' else np.nan)
# # print(df['os_event_ind'].value_counts())
# df['Meso_type'] = df['type'].apply(lambda x: 0 if x=='Epithelioid' else 1 if x=='Biphasic' else 2)
# # print(df['Meso_type'].value_counts())
# #save the dataframe to csv
# df.to_csv('%sHistomorphological-Phenotype-Learning/files/TCGA_files/clinical_TCGA_clean.csv'%main_path, index=False)


            


# get the list of slides from the text file
# slides = df['filename'].apply(lambda x: str(x.split('.')[0]).split('-')[0]+'-'+str(x.split('.')[0]).split('-')[1]+'-'+str(x.split('.')[0]).split('-')[2] ).tolist()
# print(len(set(slides)))
# print(set(slides))


h5_complete_path   = '%sHistomorphological-Phenotype-Learning/results/BarlowTwins_3/TCGA_MESO/h224_w224_n3_zdim128/hdf5_TCGA_MESO_he_combined_metadata.h5' % main_path
# # h5_complete_path = '%s/results/BarlowTwins_3/lattice_tma/h224_w224_n3_zdim128/hdf5_lattice_tma_he_test.h5'%main_path
# # # h5_complete_path = '%s/datasets/lattice_tma/he/patches_h224_w224/hdf5_lattice_tma_he_test.h5'%main_path
# # # 
file_h5_complete = h5py.File(h5_complete_path, 'r')
for key in file_h5_complete.keys():
    print('--------------------------------------------------')
    print(key)
    print(file_h5_complete[key].shape)
    print(file_h5_complete[key][-1])

# #  get the list of slides with key 'samples'
# samples = file_h5_complete['slides'][:].astype(str)
# print(len(set(samples.tolist())))
# print(set(samples.tolist()))

# read an excel file and change the name of columns. then change the elements of first column and save to csv
# excel_path = '%sHistomorphological-Phenotype-Learning/files/Patient_TCGA.xlsx' % main_path
# df = pd.read_excel(excel_path, sheet_name='1B_MPM_Master_Patient_Table')
# df['TCGA_barcode'] = df['TCGA_barcode'].apply(lambda x: x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2])
# print(df)

# find the difference between two lists
# print(set(df['TCGA_barcode'].tolist()) - set(slides))
# print(set(slides) - set(df['TCGA_barcode'].tolist()))






# csv_path = '%s/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/meso_nn250/adatas'%main_path
# resolution = 2.0
# fold = 0

# df = pd.read_csv('{}/lattice_tma_he_complete_leiden_{}__fold{}.csv'.format(csv_path, str(resolution).replace('.', 'p') ,fold))

# df['tma_slides'] = df['slides'].apply(lambda x: x.split('_')[0]+'_'+x.split('_')[1]+'_'+x.split('_')[2]+'_'+x.split('_')[3])
# print('Number of clusters: ', df['leiden_'+str(resolution)].nunique(), '\n', sorted(df['leiden_'+str(resolution)].unique()))
# df = df.groupby(['tma_slides'])['leiden_'+str(resolution)].value_counts().to_frame('frequency').reset_index()

# plt.figure(figsize=(20, 10))
# for tma_slide in df['tma_slides'].unique():
#     freq_list = df[df['tma_slides']==tma_slide]['frequency'].values
#     plt.subplot(2, 4, df['tma_slides'].unique().tolist().index(tma_slide)+1)
#     plt.bar(range(len(freq_list)), freq_list, color='c' )
#     plt.title('Leiden clusters frequency for {}'.format(tma_slide))
#     plt.xlabel('Leiden clusters')
#     plt.ylabel('Frequency')
#     plt.xticks(range(len(freq_list)), df[df['tma_slides']==tma_slide]['leiden_'+str(resolution)].values)
#     plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig('{}/leiden_clusters_frequency_fold{}.png'.format(csv_path, fold))    
# plt.close()