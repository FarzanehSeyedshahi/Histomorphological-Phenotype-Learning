import h5py
# import scanpy as sc
import numpy as np
import pandas as pd

main_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning'
h5_complete_path   = '%s/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/hdf5_Meso_250_subsampled_he_complete.h5' % main_path
csv_complete_path   = '%s/csv_Meso_250_subsampled_he_complete.csv' % main_path

# def get_folds_bins(frame, n_bins=5, eps=1e-6, num_folds=5):
#     def get_folds_event_data(frame, k):
#         # Copy.
#         df = frame.copy(deep=True)
#         # df = df.reindex(np.random.permutation(df.index)).sort_values(event_col)
#         n, _ = df.shape

#         # Fold assigments for each row entry.
#         assignments = np.array((n // k + 1) * list(range(1, k + 1)))
#         print('assignments', assignments)
#         assignments = assignments[:n]

#         # Get fold patients.
#         folds = list()
#         for i in range(1, k+1):
#             ix = assignments == i
#             training_data = df.loc[~ix]
#             test_data     = df.loc[ix]
#             training_pat  = pd.unique(training_data.case_submitter_id).tolist()
#             test_pat  = pd.unique(test_data.case_submitter_id).tolist()
#             folds.append((training_pat,test_pat))
#         return folds

#     frame_working = frame.copy(deep=True)
#     uncensored_df = frame_working[frame_working.event_ind==1]

#     disc_labels, q_bins = pd.qcut(uncensored_df['event_data'], q=n_bins, retbins=True, labels=False)
#     q_bins[-1] = frame_working['event_data'].max() + eps
#     q_bins[0] = frame_working['event_data'].min() - eps

#     disc_labels, q_bins = pd.cut(frame_working['event_data'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
#     frame_working.insert(2, 'label', disc_labels.values.astype(int))

#     total_folds = dict()
#     for i in range(num_folds):
#         total_folds[i] = dict()
#         total_folds[i]['train'] = list()
#         total_folds[i]['valid'] = list()
#         total_folds[i]['test'] = list()

#     for i in range(len(q_bins)-1):
#         bin_censored   = frame_working[(frame_working.label==i)&(frame_working.censored==1)]
#         bin_uncensored = frame_working[(frame_working.label==i)&(frame_working.censored==0)]
#         bin_folds_censored   = get_folds_event_data(frame=bin_censored,   k=num_folds)
#         bin_folds_uncensored = get_folds_event_data(frame=bin_uncensored, k=num_folds)

#         for i in range(num_folds):
#             total_folds[i]['train'].extend([(pat, None, None) for pat in bin_folds_censored[i][0]] + [(pat, None, None) for pat in bin_folds_uncensored[i][0]])
#             total_folds[i]['test'].extend([(pat, None, None) for pat in bin_folds_censored[i][1]] + [(pat, None, None) for pat in bin_folds_uncensored[i][1]])
#     return total_folds

# # Verify: This should all be empty.
# def sanity_check_overlap(folds, num_folds):
#     print('Running sanity check, this should return no samples!')
#     flag_good = True
#     # For each fold, no overlap between cells.
#     for i in range(num_folds):
#         if 'valid' in folds[i].keys():
#             result = set(folds[i]['train']).intersection(set(folds[i]['valid']))
#             if len(result) > 0:
#                 flag_good = False
#                 print(result)

#             result = set(folds[i]['valid']).intersection(set(folds[i]['test']))
#             if len(result) > 0:
#                 flag_good = False
#                 print(result)

#         result = set(folds[i]['train']).intersection(set(folds[i]['test']))
#         if len(result) > 0:
#             flag_good = False
#             print(result)

#         # No overlap between test sets of all folds.
#         for i in range(num_folds):
#             for j in range(num_folds):
#                 if i==j: continue
#                 result = set(folds[i]['test']).intersection(set(folds[j]['test']))
#                 if len(result) > 0:
#                     flag_good = False
#                     print('Fold %s-%s' % (i,j), result)
#     if flag_good:
#         print('All good')
#     else:
#         print('Review folds')
#     print('')
#     return flag_good

# def fold_samples_to_slides(folds, frame):
#     folds_slides = dict()
#     for i in folds:
#         train_samples = [sample for sample, _, _ in folds[i]['train']]
#         valid_samples = [sample for sample, _, _ in folds[i]['valid']]
#         test_samples  = [sample for sample, _, _ in folds[i]['test']]
#         train_slides = np.unique(frame[frame.samples.isin(train_samples)]['slides'].values.tolist())
#         valid_slides = np.unique(frame[frame.samples.isin(valid_samples)]['slides'].values.tolist())
#         test_slides  = np.unique(frame[frame.samples.isin(test_samples)]['slides'].values.tolist())
#         folds_slides[i] = dict()
#         folds_slides[i]['train'] = [(slide, None, None) for slide in train_slides]
#         folds_slides[i]['valid'] = [(slide, None, None) for slide in valid_slides]
#         folds_slides[i]['test']  = [(slide, None, None) for slide in test_slides]
#     return folds_slides





# import pandas as pd
# import numpy as np
# import pickle
# import copy
# import math
# import os

# def store_data(data, file_path):
#     with open(file_path, 'wb') as file:
#         pickle.dump(data, file)

# def get_frac_split(meta_df, matching_field, ind_column, num_folds=5):
#     # Copy dataframe.
#     df = meta_df.copy(deep=True)

#     # Get unique classes.
#     unique_classes = np.unique(meta_df[ind_column])
#     # randomize rows
#     df = df.sample(frac=1).reset_index(drop=True)

#     folds          = dict()
#     for i in range(num_folds):
#         folds[i] = dict()
#         folds[i]['train'] = list()
#         folds[i]['test']  = list()

#     for class_ in unique_classes:
#         # Get slides for class.
#         slides      = np.unique(df[df[ind_column]==class_][matching_field].values)

#         # Test size.
#         num_samples = len(slides)
#         test_size   = math.floor(num_samples*(1/num_folds))

#         # Iterate through chunks and add samples to fold.
#         for i in range(num_folds):
#             test_sample  = slides[i*test_size:(i+1)*test_size]
#             train_sample = list(set(slides).difference(set(test_sample)))
#             folds[i]['train'].extend(train_sample)
#             folds[i]['test'].extend(test_sample)

#     return folds

# def get_folds(meta_df, matching_field, ind_column, num_folds=5, valid_set=False):

#     # Get initial split for train/test.
#     folds = get_frac_split(meta_df, matching_field, ind_column, num_folds=num_folds)

#     for i in range(num_folds):
#         whole_train_samples = folds[i]['train']
#         subset_df = meta_df[meta_df[matching_field].isin(whole_train_samples)]
#         train_val_folds = get_frac_split(subset_df, matching_field, ind_column, num_folds=num_folds)
#         del folds[i]['train']
#         folds[i]['train'] = train_val_folds[0]['train']
#         folds[i]['valid'] = train_val_folds[0]['test']

#     return folds

# # Verify: This should all be empty.
# def sanity_check_overlap(folds, num_folds):
#     # For each fold, no overlap between cells.
#     for i in range(num_folds):
#         result = set(folds[i]['train']).intersection(set(folds[i]['valid']))
#         if len(result) > 0:
#             print(result)

#         result = set(folds[i]['train']).intersection(set(folds[i]['test']))
#         if len(result) > 0:
#             print(result)

#         result = set(folds[i]['valid']).intersection(set(folds[i]['test']))
#         if len(result) > 0:
#             print(result)

#         # No overlap between test sets of all folds.
#         for i in range(num_folds):
#             for j in range(num_folds):
#                 if i==j: continue
#                 result = set(folds[i]['test']).intersection(set(folds[j]['test']))
#                 if len(result) > 0:
#                     print('Fold %s-%s' % (i,j), result)

# # Fit for legacy code.
# def fit_format(folds):
#     slides_folds = dict()
#     for i, fold in enumerate(folds):
#         slides_folds[i] = dict()
#         slides_folds[i]['train'] = [(slide, None, None) for slide in folds[i]['train']]
#         slides_folds[i]['valid'] = [(slide, None, None) for slide in folds[i]['valid']]
#         slides_folds[i]['test']  = [(slide, None, None) for slide in folds[i]['test']]

#     return slides_folds

#reading h5 file
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_h5(file_path):
    with h5py.File(file_path, 'r') as file:
        for key in file.keys():
            print(key)
            print(file[key].shape)
            print(file[key][0])
            print('--------------------------------------------------')
        extracted_rows = {}
        mask = file['slides'][:] == b'MESO_406_6'
        for column_name in file.keys():
            column_data = file[column_name][:]
            extracted_rows[column_name] = column_data[mask]
        print(extracted_rows)
    return extracted_rows

# data = read_h5('%s/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/hdf5_Meso_250_subsampled_he_complete.h5' % main_path)

def making_csv(file_path):
    with h5py.File(h5_complete_path) as df:
        # patient_csv = pd.read_excel('files/Mesotheliomanovember_DATA_LABELS_2020-07-31_1737.xlsx')
        # patient_csv = patient_csv[['Case Number', 'Slides to De-archive', 'Blocks to De-archive', 'Data Source', \
        #                            'Operation', 'Side', 'Mesothelioma Type', 'Desmoplastic Component', 'Diaphragm Involvement', \
        #                             'Rib Involvement', 'Lung Involvement', 'Chest Wall Involvement', 'N Stage', 'T Stage', \
        #                                 'M Stage', 'Overall Stage (7th Edition TNM)', 'Overall Stage (8th Edition TNM)', \
        #                                    'Confident Diagnosis of Mesothelioma?', 'Survival Status',  'Time to Survival Status (Days)'\
        #                                        , 'Time to Survival Status (Months)', 'Hb score', 'Haemoglobin Measurement (g/dL)', 'Core Mesothelioma Type'  ]]
        # patient_csv.to_csv('%s/files/patient_csv_fitered.csv' % main_path)
        filtered_patient_csv = pd.read_csv('%s/files/patient_csv_fitered.csv' % main_path)
        filtered_patient_csv = filtered_patient_csv[['Case Number', 'Time to Survival Status (Months)', 'Time to Survival Status (Days)', 'Core Mesothelioma Type', \
                                                     'Overall Stage (8th Edition TNM)', 'Survival Status']]
        filtered_patient_csv = filtered_patient_csv.rename(columns={'Case Number': 'case_number', 'Time to Survival Status (Months)': 'survival_months', \
                                                                    'Time to Survival Status (Days)': 'survival_days', 'Core Mesothelioma Type': 'Meso_type', \
                                                                        'Overall Stage (8th Edition TNM)': 'stage', 'Survival Status': 'survival_status'})
        
        filtered_patient_csv['case_number'] = filtered_patient_csv['case_number'][:].astype(int)
        print(filtered_patient_csv.info())

        slides = df['slides'][:].astype(str)
        samples = ['_'.join(slide.split('_')[:2]) for slide in slides]
        case_number = [slide.split('_')[1] for slide in slides]
        final_df = pd.DataFrame({'case_number': case_number, 'slides': slides, 'samples': samples, 'hist_subtype': df['hist_subtype'][:].astype(str), 'tiles': df['tiles'][:].astype(str)})
        final_df['case_number'] = final_df['case_number'][:].astype(int)
        print(final_df.info())
        # final_df.to_csv('%s/files/csv_Meso_250_subsampled_he_complete.csv' % main_path)

        merged_df = pd.merge(final_df, filtered_patient_csv, on='case_number', how='left')
        print(merged_df.info())
        merged_df.to_csv('%s/files/csv_Meso_250_subsampled_he_complete_merged.csv' % main_path)

making_csv(h5_complete_path)