main_path = '/mnt/cephfs/sharedscratch/users/fshahi/Projects/Histomorphological-Phenotype-Learning'
mil_path = '{}/workflow/mil'.format(main_path)
import sys
sys.path.append(main_path)
from models.clustering.data_processing import *
import numpy as np
import pickle
from tqdm import tqdm

dataset = 'Meso'
h5_complete_path   = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/hdf5_{}_he_complete_filtered_metadata.h5'.format(main_path, dataset, dataset)
frame, dims, rest = representations_to_frame(h5_complete_path, meta_field='Meso_type', rep_key='z_latent')


additional_dataset = 'TCGA_MESO'
h5_additional_path = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/hdf5_{}_he_complete_filtered_metadata.h5'.format(main_path, additional_dataset, additional_dataset)
frame_additional, dims_additional, rest_additional = representations_to_frame(h5_additional_path, meta_field='Meso_type', rep_key='z_latent')

frame.replace("-", np.inf, inplace=True)
frame_additional.replace("'--", np.inf, inplace=True)
frame_additional['os_event_data'] = frame_additional['os_event_data'].apply(lambda x: round(float(x)/30.0,2) if x != np.inf else x)

pkl_file = open('/nfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning/files/pkl_Meso_he_test_train_cases.pkl', 'rb')
train_test_cases = pickle.load(pkl_file)
csv_data = frame.iloc[:,0:128]
csv_data['Meso_type'] = frame['Meso_type'].astype(float)
csv_data['slides'] = frame['slides']
csv_data['case_Id'] = frame['case_Id'].astype(str)
csv_data['os_event_ind'] = frame['os_event_ind'].astype(float)
csv_data['os_event_data'] = frame['os_event_data'].astype(float)
csv_data.replace(np.inf, 15*15, inplace=True)


csv_data_add = frame_additional.iloc[:,0:128]
# map the Meso_type to 0(0), 1,2(1)
csv_data_add['Meso_type'] = frame_additional['Meso_type'].replace({'0': '0', '1': '1', '2': '1'})
csv_data_add['Meso_type'] = csv_data_add['Meso_type'].astype(float)
csv_data_add['slides'] = frame_additional['slides']
csv_data_add['case_Id'] = frame_additional['case_Id']
csv_data_add['os_event_ind'] = frame_additional['os_event_ind'].astype(float)
csv_data_add['os_event_data'] = frame_additional['os_event_data'].astype(float)
csv_data_add.replace(np.inf, 15*15, inplace=True)



for fold in tqdm(range(5)):
    train = train_test_cases[fold]['train']
    test = train_test_cases[fold]['test']
    train_cases = np.array([x[0] for x in train]).astype(str)
    test_cases = np.array([x[0] for x in test]).astype(str)
    train_df = csv_data[csv_data['case_Id'].isin(train_cases)]
    test_df = csv_data[csv_data['case_Id'].isin(test_cases)]
    val_df = csv_data_add
    print('train_df shape:', train_df.shape)
    print('test_df shape:', test_df.shape)
    print('val_df shape:', val_df.shape)


    # subtype npys
    data_list = []
    for slide in train_df['slides'].unique():
        instance = train_df[train_df['slides'] == slide].iloc[:,0:128].values
        label = train_df[train_df['slides'] == slide]['Meso_type'].values[0]
        data_list.append([instance, label, slide])
    data_list = np.array(data_list)
    np.save('{}/data/subtype_train_fold_{}.npy'.format(mil_path,fold), data_list)
    data_list = []
    for slide in test_df['slides'].unique():
        instance = test_df[test_df['slides'] == slide].iloc[:,0:128].values
        label = test_df[test_df['slides'] == slide]['Meso_type'].values[0]
        data_list.append([instance, label, slide])
    data_list = np.array(data_list)
    np.save('{}/data/subtype_test_fold_{}.npy'.format(mil_path,fold), data_list)
    # additional dataset npys
    data_list = []
    for slide in val_df['slides'].unique():
        instance = val_df[val_df['slides'] == slide].iloc[:,0:128].values
        label = val_df[val_df['slides'] == slide]['Meso_type'].values[0]
        data_list.append([instance, label, slide])
    data_list = np.array(data_list)
    np.save('{}/data/subtype_additional_fold_{}.npy'.format(mil_path,fold), data_list)




    # survival npys
    data_list = []
    for case in train_df['case_Id'].unique():
        instance = train_df[train_df['case_Id'] == case].iloc[:,0:128].values
        ind = train_df[train_df['case_Id'] == case]['os_event_ind'].values[0]
        data = train_df[train_df['case_Id'] == case]['os_event_data'].values[0]
        data_list.append([instance, ind, data, case])
    data_list = np.array(data_list)
    np.save('{}/data/survival_train_fold_{}.npy'.format(mil_path,fold), data_list)
    data_list = []
    for case in test_df['case_Id'].unique():
        instance = test_df[test_df['case_Id'] == case].iloc[:,0:128].values
        ind = test_df[test_df['case_Id'] == case]['os_event_ind'].values[0]
        data = test_df[test_df['case_Id'] == case]['os_event_data'].values[0]
        data_list.append([instance, ind, data, case])
    data_list = np.array(data_list)
    np.save('{}/data/survival_test_fold_{}.npy'.format(mil_path,fold), data_list)
    # additional dataset npys
    data_list = []
    for case in val_df['case_Id'].unique():
        instance = val_df[val_df['case_Id'] == case].iloc[:,0:128].values
        ind = val_df[val_df['case_Id'] == case]['os_event_ind'].values[0]
        data = val_df[val_df['case_Id'] == case]['os_event_data'].values[0]
        data_list.append([instance, ind, data, case])
    data_list = np.array(data_list)
    np.save('{}/data/survival_additional_fold_{}.npy'.format(mil_path,fold), data_list)
        
    