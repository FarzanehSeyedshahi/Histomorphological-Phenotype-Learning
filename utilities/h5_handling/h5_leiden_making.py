import h5py
import pandas as pd

# Print the structure of an h5 file
def print_h5_structure(h5_file):
    with h5py.File(h5_file, 'r') as h5_file:
        print(h5_file.keys())
        for key in h5_file.keys():
            print(key)
            print(h5_file[key][0])
            print(h5_file[key].shape)
            print('------------------')


if __name__ == '__main__':
    data_dir = "/raid/users/farzaneh/Histomorphological-Phenotype-Learning/"
    h5_file_path = '{}results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/hdf5_Meso_250_subsampled_he_complete_combined_metadata_filtered.h5'.format(data_dir)
    csv_file_path = '{}files/average_cores_latent.csv'.format(data_dir)
    h5_core_file_path = '{}results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/hdf5_representation_cores.h5'.format(data_dir)

    print_h5_structure(h5_file_path)    

    df = pd.read_csv(csv_file_path)
    # remove h5 cores file if exists
    import os
    if os.path.exists(h5_core_file_path):
        os.remove(h5_core_file_path)

    with h5py.File(h5_core_file_path, 'w') as h5_cores_file:
            h5_cores_file.create_dataset('img_z_latent', data=df.iloc[:, 1:129].values)
            h5_cores_file.create_dataset('slides', data=df['slides'].values)
    print_h5_structure(h5_core_file_path)
