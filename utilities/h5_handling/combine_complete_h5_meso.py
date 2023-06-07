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
        

if __name__=='__main__':
    
    main_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning'
    h5_path   = '%s/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/hdf5_Meso_250_subsampled_he_complete.h5' % main_path
    csv_metadata_path = '%s/files/csv_Meso_250_subsampled_he_complete_with_survival.csv' % main_path
    h5_complete_path = '%s/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/hdf5_Meso_250_subsampled_he_complete_combined_metadata.h5'%main_path


    combine_h5(h5_path, csv_metadata_path, h5_complete_path)
