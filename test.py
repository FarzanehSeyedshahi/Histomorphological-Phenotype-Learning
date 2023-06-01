import h5py
# import scanpy as sc
import numpy as np
import pandas as pd
# df = h5py.File('data/hdf5_Meso_test.h5', 'r')
# keys_arr = ['hist_subtype', 'img_h_latent', 'img_z_latent', 'labels', 'patterns', 'samples', 'slides', 'tiles']
# keys_arr = ['img_z_latent', 'samples', 'slides', 'tiles']
# f_latent = pd.DataFrame(df_latent['img_z_latent'][:].tolist())
# f_latent['Core'] = df_latent['samples'][:]
# f_latent['Core'] = f_latent['Core'].apply(lambda x: x.decode('utf-8')[:-6])
# f_latent['tiles'] = df_latent['tiles'][:]
# f_latent['tiles'] = f_latent['tiles'].apply(lambda x: x.decode('utf-8'))
# f_latent = f_latent
# f_HPC = pd.read_csv('data/clusters_filtered.csv')[['Core', 'tiles', 'leiden_2.0']]
# merged = pd.merge(f_HPC, f_latent, on=['Core', 'tiles'])
# print(merged.columns)
# Print the number of same and different core tiles
# np.sum(core_tiles_latent, core_tiles_HPC)
# adata = sc.read_h5ad('TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered_leiden_2p0__fold0_subsample.h5ad')
# representations = adata.X
# HPCs = adata.obs['leiden_2.0']
# HPCs_long = HPCs.values.__array__(dtype = np.int_)
# np.savez('mydata.npz', representations = representations, HPC = HPCs_long)
# data = np.load('mydata.npz')
# print(data['representations'].shape)
## or csv
# df = pd.DataFrame({'representations' : representations, 'HPC' : HPCs_long})
# print(df.info())
# df.to_csv('mydata.csv')
# df2 = pd.read_csv('mydata.csv')
# print(df.info())
# # Count the number of samples per class
# for col in df2.columns:
#     counts = df2['{}'.format(col)].value_counts()
#     # Print the counts for each class
#     print(counts)
main_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning'
h5_complete_path   = '%s/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/hdf5_Meso_250_subsampled_he_complete.h5' % main_path
# with h5py.File(h5_complete_path) as df:
#     for key in df.keys():
#         print('key', key)
#         print('shape', df[key].shape )
#         print('dtype', df[key].dtype)
#         print('first', df[key][0])
#         print('first shape', np.shape(df[key][0]))
#         print('-----------------------------------')
#     print('-----------------------------------')
#     print('-----------------------------------')
#     slides = df['slides'][:].astype(str)
#     print('slides', slides[:10] )
#     samples = ['_'.join(slide.split('_')[:2]) for slide in slides]
#     print('samples', samples[:10] )
#     final_df = pd.DataFrame({'slides': slides, 'samples': samples, 'hist_subtype': df['hist_subtype'][:].astype(str), 'tiles': df['tiles'][:].astype(str)})
#     print(final_df.info())
#     print(final_df.head())
#     final_df.to_csv('%s/csv_Meso_250_subsampled_he_complete.csv' % main_path)
csv_complete_path   = '%s/csv_Meso_250_subsampled_he_complete.csv' % main_path

def get_folds_bins(frame, n_bins=5, eps=1e-6, num_folds=5):
    def get_folds_event_data(frame, k):
        # Copy.
        df = frame.copy(deep=True)
        # df = df.reindex(np.random.permutation(df.index)).sort_values(event_col)
        n, _ = df.shape

        # Fold assigments for each row entry.
        assignments = np.array((n // k + 1) * list(range(1, k + 1)))
        print('assignments', assignments)
        assignments = assignments[:n]

        # Get fold patients.
        folds = list()
        for i in range(1, k+1):
            ix = assignments == i
            training_data = df.loc[~ix]
            test_data     = df.loc[ix]
            training_pat  = pd.unique(training_data.case_submitter_id).tolist()
            test_pat  = pd.unique(test_data.case_submitter_id).tolist()
            folds.append((training_pat,test_pat))
        return folds

    frame_working = frame.copy(deep=True)
    uncensored_df = frame_working[frame_working.event_ind==1]

    disc_labels, q_bins = pd.qcut(uncensored_df['event_data'], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = frame_working['event_data'].max() + eps
    q_bins[0] = frame_working['event_data'].min() - eps

    disc_labels, q_bins = pd.cut(frame_working['event_data'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    frame_working.insert(2, 'label', disc_labels.values.astype(int))

    total_folds = dict()
    for i in range(num_folds):
        total_folds[i] = dict()
        total_folds[i]['train'] = list()
        total_folds[i]['valid'] = list()
        total_folds[i]['test'] = list()

    for i in range(len(q_bins)-1):
        bin_censored   = frame_working[(frame_working.label==i)&(frame_working.censored==1)]
        bin_uncensored = frame_working[(frame_working.label==i)&(frame_working.censored==0)]
        bin_folds_censored   = get_folds_event_data(frame=bin_censored,   k=num_folds)
        bin_folds_uncensored = get_folds_event_data(frame=bin_uncensored, k=num_folds)

        for i in range(num_folds):
            total_folds[i]['train'].extend([(pat, None, None) for pat in bin_folds_censored[i][0]] + [(pat, None, None) for pat in bin_folds_uncensored[i][0]])
            total_folds[i]['test'].extend([(pat, None, None) for pat in bin_folds_censored[i][1]] + [(pat, None, None) for pat in bin_folds_uncensored[i][1]])
    return total_folds

# Verify: This should all be empty.
def sanity_check_overlap(folds, num_folds):
    print('Running sanity check, this should return no samples!')
    flag_good = True
    # For each fold, no overlap between cells.
    for i in range(num_folds):
        if 'valid' in folds[i].keys():
            result = set(folds[i]['train']).intersection(set(folds[i]['valid']))
            if len(result) > 0:
                flag_good = False
                print(result)

            result = set(folds[i]['valid']).intersection(set(folds[i]['test']))
            if len(result) > 0:
                flag_good = False
                print(result)

        result = set(folds[i]['train']).intersection(set(folds[i]['test']))
        if len(result) > 0:
            flag_good = False
            print(result)

        # No overlap between test sets of all folds.
        for i in range(num_folds):
            for j in range(num_folds):
                if i==j: continue
                result = set(folds[i]['test']).intersection(set(folds[j]['test']))
                if len(result) > 0:
                    flag_good = False
                    print('Fold %s-%s' % (i,j), result)
    if flag_good:
        print('All good')
    else:
        print('Review folds')
    print('')
    return flag_good

def fold_samples_to_slides(folds, frame):
    folds_slides = dict()
    for i in folds:
        train_samples = [sample for sample, _, _ in folds[i]['train']]
        valid_samples = [sample for sample, _, _ in folds[i]['valid']]
        test_samples  = [sample for sample, _, _ in folds[i]['test']]
        train_slides = np.unique(frame[frame.samples.isin(train_samples)]['slides'].values.tolist())
        valid_slides = np.unique(frame[frame.samples.isin(valid_samples)]['slides'].values.tolist())
        test_slides  = np.unique(frame[frame.samples.isin(test_samples)]['slides'].values.tolist())
        folds_slides[i] = dict()
        folds_slides[i]['train'] = [(slide, None, None) for slide in train_slides]
        folds_slides[i]['valid'] = [(slide, None, None) for slide in valid_slides]
        folds_slides[i]['test']  = [(slide, None, None) for slide in test_slides]
    return folds_slides




# import pandas as pd
# import os

# # Set the path to the folder containing the CSV files
# folder_path = 'adatas/'

# # Get a list of all CSV files in the folder
# file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv') & ('1p0' in file) ]

# # Initialize an empty dataframe
# df = pd.DataFrame()

# # Loop through each CSV file and append it to the dataframe
# for file in file_list:
#     file_path = os.path.join(folder_path, file)
#     temp_df = pd.read_csv(file_path)
#     print(file)
#     print(temp_df.info())
#     df = df._append(temp_df)

# print('final df+++++++++++/n', df.info())
# # # Reset the index of the dataframe
# # df = df.reset_index(drop=True)
# subset_ind = df['leiden_1.0'].values
# print(subset_ind, len(set(subset_ind)))





# f = h5py.File('hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5', 'r')
# tot_ind = f['indexes']

# import numpy as np
# indexes = np.where(np.isin(tot_ind, subset_ind))[0]
# print(indexes, len(indexes))
# unique_values, counts = np.unique(tot_ind, return_counts=True)
# print(len(unique_values), len(unique_values[counts > 1]))
# print(df[df['indexes']==7288])
# Create the numpy array with repeated elements
# arr = np.array([1, 2,3, 2, 3,5,  3, 3, 4, 4, 4])

# # Find the unique values and their counts
# unique_values, counts = np.unique(arr, return_counts=True)
# print(unique_values, counts)
# # Print the repeated values
# repeated_values = unique_values[counts > 1]
# print(repeated_values)