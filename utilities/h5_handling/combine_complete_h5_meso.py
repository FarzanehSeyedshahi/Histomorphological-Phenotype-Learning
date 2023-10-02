import h5py
import numpy as np
import pandas as pd
import os

# A function for loading the h5 file in pandas dataframe and merge the metadata csv file using pandas
def combine_h5(file_path, metadata_path, h5_complete_path, override=True):
    
    
    # delete the file if it exists
    if override:
        if os.path.exists(h5_complete_path):
            os.remove(h5_complete_path)

    
    content = h5py.File(h5_complete_path, mode='a')
    with h5py.File(file_path, 'r') as file:
        metadata = pd.read_csv(metadata_path, index_col=0)        
        for column_name in file.keys():
            if column_name == 'slides':
                    # Sanity check to see if the slides are the same in the h5 file and the metadata csv file
                    slides_np_h5 = np.array(file[column_name][:].astype(str))
                    slides_np_csv = metadata['slides'].to_numpy()
        
        if (slides_np_h5==slides_np_csv).all():
            for column_name in file.keys():
                content.create_dataset(name=column_name, data=file[column_name][:])
                print(column_name, 'created.')
            print('=> h5 file done')
            for column_name in metadata.columns:
                #check the column name if already exists in the h5 file
                if column_name in content.keys():
                    print(column_name, 'already exists!')
                    continue
                else:
                    content.create_dataset(name=column_name, data=metadata[column_name][:].to_numpy())
                    print(column_name, 'created.')

            print('=> metadata done.')
        else:
            print('The slides order are not similar!')
            return
    
    print('\n------------------------------ New Generated h5 file:')
    for key in content.keys():
        print('--------------------------------------------------')
        print(key)
        print(content[key].shape)
        print(content[key][-1])

def add_Meso_type_column(h5_path, h5_complete_path, override=True):
    # delete the file if it exists
    if override:
        if os.path.exists(h5_complete_path):
            os.remove(h5_complete_path)

    content = h5py.File(h5_complete_path, mode='a')
    with h5py.File(h5_path, 'r') as file:

        for column_name in file.keys():
            temp = file[column_name][:]
            #check if any nan value exists in the column
            if np.issubdtype(file[column_name][:].dtype, np.number):
                if np.isnan(file[column_name][:]).any():
                    print(column_name, 'has nan values!')
                    # file[column_name][:][np.isnan(file[column_name][:])] = 0.0
                    # replace the nan values with zero
                    temp = np.nan_to_num(file[column_name][:])
            content.create_dataset(name=column_name, data=temp)
            # Add the Meso type column
            print(column_name, 'created.')

            if column_name == 'type' and 'Meso_type' not in content.keys():
                    # finding the unique values
                    # Creating a column that put zero if the type is Epithelioid and 1 if the type is Sarcomatoid or Biphasic
                    # 0: Epithelioid    1: Sarcomatoid or Biphasic
                    type = file[column_name][:]
                    print(type)
                    type = np.where(type == b'Epithelioid', 0, 1)
                    print(type)
                    print(type.shape)
                    # print the number of each type
                    content.create_dataset(name='Meso_type', data=type)
                    print('Meso_type created.')
        
    
    print('\n------------------------------ New Generated h5 file:')
    for key in content.keys():
        print('--------------------------------------------------')
        print(key)
        print(content[key].shape)
        print(content[key][-1])
        

if __name__=='__main__':
    
    main_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning'
    h5_path   = '%s/results/BarlowTwins_3/Meso_400_subsampled/h224_w224_n3_zdim128/hdf5_Meso_400_subsampled_he_complete_filtered.h5' % main_path
    csv_metadata_path = '%s/files/csv_Meso_400_subsampled_he_complete_filtered_with_survival.csv' % main_path
    # h5_complete_path = '%s/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/hdf5_Meso_250_subsampled_he_complete_combined_metadata.h5'%main_path
    h5_complete_path = '%s/results/BarlowTwins_3/Meso_400_subsampled/h224_w224_n3_zdim128/hdf5_Meso_400_subsampled_he_complete_filtered_combined_metadata.h5'%main_path

    combine_h5(h5_path, csv_metadata_path, h5_complete_path)
    # add_Meso_type_column(h5_path, h5_complete_path)
