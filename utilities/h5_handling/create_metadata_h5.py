# Imports.
import pandas as pd
import numpy as np
import argparse
import h5py
import sys
import os

# Add project path
main_path = os.path.dirname(os.path.realpath(__file__))
main_path = '/'.join(main_path.split('/')[:-2])
sys.path.append(main_path)
from tqdm import tqdm

# Folder permissions for cluster.
os.umask(0o002)
# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

##### Methods #######
# Read metada data and list of individuals.
def read_meta_data(file_path, matching_field):
    frame = pd.read_csv(file_path)
    meta_individuals = list(sorted(frame[matching_field].values.astype(str)))
    return frame, meta_individuals


# Get number of samples that overlap with individuals in the meta file.
def h5_overlap_meta_individuals(h5_file, matching_field, meta_individuals):
    h5_samples = 0
    print('H5 File: %s' % h5_file)
    with h5py.File(h5_file, 'r') as content:
        h5_individual_prev = ''
        match_flag         = False
        for index_h5 in tqdm(range(content[matching_field].shape[0])):
            h5_individual = content[matching_field][index_h5].decode("utf-8")
            if h5_individual != h5_individual_prev:
                if h5_individual in meta_individuals:
                    match_flag = True
                else:
                    match_flag = False
                h5_individual_prev = h5_individual
            if match_flag:
                h5_samples +=1

    return h5_samples

# Get key_names, shape, and dtype
def data_specs(h5_path):
    key_dict = dict()
    with h5py.File(h5_path, 'r') as content:
        for key in content.keys():
            key_dict[key] = dict()
            key_dict[key]['shape'] = content[key].shape[1:]
            key_dict[key]['dtype'] = content[key].dtype

    return key_dict

# Create new H5 file with individuals in the meta file.
def create_metadata_h5(h5_file, meta_name, list_meta_field, matching_field, meta_individuals, num_tiles, key_dict, override):
    h5_metadata_path = h5_file.replace('.h5', '_%s.h5' % meta_name)
    print('-------------------------------- Creating H5 file: %s' % h5_metadata_path)

    if os.path.isfile(h5_metadata_path):
        if override:
            os.remove(h5_metadata_path)
        else:
            print('File already exists, if you want to overwrite enable the flag --override')
            print(h5_metadata_path)
            print()
            exit()

    storage_dict = dict()
    content      = h5py.File(h5_metadata_path, mode='w')
    for key in key_dict:
        print('\tCreating dataset: %s' % key)
        shape = [num_tiles] + list(key_dict[key]['shape'])
        print('\t\tShape: %s' % shape)
        dtype = key_dict[key]['dtype']
        print('\t\tDtype: %s' % dtype)
        storage_dict[key] = content.create_dataset(name=key.replace('train_', ''), shape=shape, dtype=dtype)
        print(storage_dict[key][0])
        print('\n--------------------------------')

    dt = h5py.special_dtype(vlen=str)
    print('dt: %s' % dt)
    for meta_field in list_meta_field:
        print('\tCreating dataset: %s' % meta_field)
        dtype = frame[meta_field].dtype
        if str(dtype) == 'object':
            dtype = dt
        storage_dict[meta_field] = content.create_dataset(name=meta_field, shape=[num_tiles], dtype=dtype)


    index = 0
    print('Iterating through %s ...' % h5_file.split('/')[-1])
    with h5py.File(h5_file, 'r') as orig_content:
        set_dict = dict()
        for key in storage_dict:
            flag_meta_field = False
            for meta_field in list_meta_field:
                if key == meta_field:
                    flag_meta_field = True
                    break
            if flag_meta_field:
                continue
            set_dict[key] = orig_content[key]

        for i in range(set_dict[list(storage_dict.keys())[0]].shape[0]):
            # Verbose
            if i%100000==0:
                print('\tprocessed %s entries' % i)

            # If sample doesn't overlap with meta file, get rid of it.
            h5_individual = set_dict[matching_field][i].decode("utf-8")
            if h5_individual not in meta_individuals:
                continue
            for key in storage_dict:
                # Check for field to include.
                if key in list_meta_field:
                    # print('storage_dict[key]', storage_dict[key])
                    # print('index', index)
                    # print('key', key)
                    # print('storage_dict[key][index] shape', np.shape(storage_dict[key][index]))
                    # print('h5_individual', h5_individual)
                    # print('frame[matching_field].astype(str)', frame[matching_field].astype(str))
                    # print('matching_field', matching_field)
                    # print('frame\n', frame[frame[matching_field].astype(str)==str(h5_individual)][key])
                    # print('frame shape', np.shape(frame[frame[matching_field].astype(str)==str(h5_individual)][key]))
                    storage_dict[key][index] = frame[frame[matching_field].astype(str)==str(h5_individual)][key]
                # Copy all other fiels
                else:
                    storage_dict[key][index] = set_dict[key][i]
                    
            index += 1
        print()


##### Main #######
parser = argparse.ArgumentParser(description='Script to create a subset H5 representation file based on meta data file.')
parser.add_argument('--meta_file',        dest='meta_file',        type=str,             required=True,  help='Path to CSV file with meta data.')
parser.add_argument('--meta_name',        dest='meta_name',        type=str,             required=True,  help='Name to use to rename H5 file.')
parser.add_argument('--list_meta_field',  dest='list_meta_field',  type=str,             required=True,  help='Field name that contains the information to include in the H5 file.', nargs="*")
parser.add_argument('--matching_field',   dest='matching_field',   type=str,             required=True,  help='Reference filed to use, cross check between original H5 and meta file.')
parser.add_argument('--h5_file',          dest='h5_file',          type=str,             required=True,  help='Original H5 file to parse.')
parser.add_argument('--override',         dest='override',         action='store_true',  default=False,  help='Override \'complete\' H5 file if it already exists.')
args            = parser.parse_args()
meta_file       = args.meta_file
meta_name       = args.meta_name
list_meta_field = args.list_meta_field
matching_field  = args.matching_field
h5_file         = args.h5_file
override        = args.override

# Read meta data file and list of individuals according to the matching_field.
frame, meta_individuals = read_meta_data(meta_file, matching_field)
print('Number of individuals in meta file: %s' % len(meta_individuals))
if matching_field == 'case_Id':
    frame[matching_field] = frame[matching_field].astype('S23')


# Get number of tiles from all individuals in the original H5, <= to the original.
num_tiles = h5_overlap_meta_individuals(h5_file, matching_field, meta_individuals)
print('Number of tiles in original H5: %s' % num_tiles)

# Dictionary with keys, shapes, and dtypes.
key_dict = data_specs(h5_file)
print('Keys in original H5: %s' % list(key_dict.keys()))

# Create H5 with the list of individuals and the field.
create_metadata_h5(h5_file, meta_name, list_meta_field, matching_field, meta_individuals, num_tiles, key_dict, override)
print('Created H5 file: %s' % h5_file.replace('.h5', '_%s.h5' % meta_name))