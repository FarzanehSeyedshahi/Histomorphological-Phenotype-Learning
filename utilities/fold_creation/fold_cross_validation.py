import os
import pandas as pd
import numpy as np
import h5py
import pickle
import math

# Workspace path.
# main_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning'
main_path = '/mnt/cephfs/sharedscratch/users/fshahi/Projects/Histomorphological-Phenotype-Learning'
# csv_path = '%s/files/patient_csv_fitered.csv' % main_path
project   = 'Meso'

def grab_slides_samples(h5_path, tcga_flag=True):
    samples = None
    with h5py.File(h5_path) as content:
        print('Content:', content.keys())
        slides = content['slides'][:].astype(str)
        if 'samples' not in content.keys():
            print('Samples not find in the H5 file. Creating from slide information.')
            if tcga_flag:
                samples = ['-'.join(slide.split('-')[:3]) for slide in slides]
        else:
            samples = content['samples'][:].astype(str)

    print()
    print('Slides: ', len(slides))
    print('Samples:', len(samples))

    df = pd.DataFrame(slides, columns=['slides'])
    df['samples'] = samples
    return  slides, samples, df

def creat_csv_from_h5_wsi(csv_results_path, h5_complete_path):
    with h5py.File(h5_complete_path) as df:
        slides = df['slides'][:].astype(str)
        samples = ['_'.join(slide.split('_')[:2]) for slide in slides]
        final_df = pd.DataFrame({'slides': slides, 'samples': samples, 'hist_subtype': df['hist_subtype'][:].astype(str), 'tiles': df['tiles'][:].astype(str)})
        final_df.to_csv(csv_results_path)
    return final_df

def creat_csv_from_h5_tma(csv_results_path, h5_complete_path):
    with h5py.File(h5_complete_path) as df:
        final_df = pd.DataFrame({'slides': df['slides'][:].astype(str), 'samples': df['samples'][:].astype(str), 'hist_subtype': df['hist_subtype'][:].astype(str), 'tiles': df['tiles'][:].astype(str)})
        final_df.to_csv(csv_results_path)
    return final_df

# reading pkl file
def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print('Data keys:', data.keys())
        for key in data.keys():
            print('-------------------------------------------------------------------')
            print('Key:', key, '- Number of train:', len(data[key]['train']), '- Number of valid:', len(data[key]['valid']), '- Number of test:', len(data[key]['test']))

    return data

def store_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def get_frac_split(meta_df, matching_field, ind_column, num_folds=5):
    # Copy dataframe.
    df = meta_df.copy(deep=True)

    # Get unique classes.
    unique_classes = np.unique(meta_df[ind_column])
    # randomize rows
    df = df.sample(frac=1).reset_index(drop=True)

    folds          = dict()
    for i in range(num_folds):
        folds[i] = dict()
        folds[i]['train'] = list()
        folds[i]['test']  = list()

    for class_ in unique_classes:
        # Get slides for class.
        slides      = np.unique(df[df[ind_column]==class_][matching_field].values)

        # Test size.
        num_samples = len(slides)
        test_size   = math.floor(num_samples*(1/num_folds))

        # Iterate through chunks and add samples to fold.
        for i in range(num_folds):
            test_sample  = slides[i*test_size:(i+1)*test_size]
            train_sample = list(set(slides).difference(set(test_sample)))
            folds[i]['train'].extend(train_sample)
            folds[i]['test'].extend(test_sample)

    return folds

def get_folds(meta_df, matching_field, ind_column, num_folds=5, valid_set=False):

    # Get initial split for train/test.
    folds = get_frac_split(meta_df, matching_field, ind_column, num_folds=num_folds)
    for i in range(num_folds):
        print('Fold %s:' % i, 'train:', len(folds[i]['train']), 'test:', len(folds[i]['test']))

    for i in range(num_folds):
        whole_train_samples = folds[i]['train']
        subset_df = meta_df[meta_df[matching_field].isin(whole_train_samples)]
        train_val_folds = get_frac_split(subset_df, matching_field, ind_column, num_folds=num_folds)
        del folds[i]['train']
        folds[i]['train'] = train_val_folds[0]['train']
        folds[i]['valid'] = train_val_folds[0]['test']

    return folds

# Verify: This should all be empty.
def sanity_check_overlap(folds, num_folds):
    # For each fold, no overlap between cells.
    for i in range(num_folds):
        result = set(folds[i]['train']).intersection(set(folds[i]['valid']))
        if len(result) > 0:
            print(result)

        result = set(folds[i]['train']).intersection(set(folds[i]['test']))
        if len(result) > 0:
            print(result)

        result = set(folds[i]['valid']).intersection(set(folds[i]['test']))
        if len(result) > 0:
            print(result)

        # No overlap between test sets of all folds.
        for i in range(num_folds):
            for j in range(num_folds):
                if i==j: continue
                result = set(folds[i]['test']).intersection(set(folds[j]['test']))
                if len(result) > 0:
                    print('Fold %s-%s' % (i,j), result)

# Fit for legacy code.
def fit_format(folds):
    slides_folds = dict()
    for i, fold in enumerate(folds):
        slides_folds[i] = dict()
        slides_folds[i]['train'] = [(slide, None, None) for slide in folds[i]['train']]
        slides_folds[i]['valid'] = [(slide, None, None) for slide in folds[i]['valid']]
        slides_folds[i]['test']  = [(slide, None, None) for slide in folds[i]['test']]

    return slides_folds

def create_csv_wsi(representation_csv, patient_csv, csv_results_path):
    '''This function get metadata file from patient(patient_csv) and tile_vector 
    presentation(representation_csv) and create a new csv file with 
    the wanted columns to combine with the h5 file as a metadata file.(csv_results_path)'''

    #delete the csv file if it exists
    if os.path.exists(csv_results_path):
        os.remove(csv_results_path)
    
    # reading csv file and get the slides and samples and change the column name as wanted
    df_patient = pd.read_csv(patient_csv, index_col=0)
    df_patient = df_patient[['Case Number', 'Mesothelioma Type', 'N Stage (8th Edition TNM)', 'Survival Status', 'Time to Survival Status (Months)', 'Smoking History', 'WCC Score', 'Desmoplastic Component']]
    df_patient = df_patient.rename(columns={'Case Number': 'case_number', \
                                            'Mesothelioma Type': 'type', \
                                                'N Stage (8th Edition TNM)':'stage',\
                                                      'Survival Status': 'os_event_ind', \
                                                        'Time to Survival Status (Months)': 'os_event_data', \
                                                             'Smoking History': 'smoking_history', \
                                                                 'WCC Score': 'wcc_score', \
                                                                     'Desmoplastic Component': 'desmoplastic_component' })
    
    # make the os_event_ind elements to binary. Dead: 1, Alive: 0
    df_patient['os_event_ind'] = df_patient['os_event_ind'].replace('Dead', 1.0)
    df_patient['os_event_ind'] = df_patient['os_event_ind'].replace('Alive', 0.0)
    df_patient['Meso_type'] = df_patient['type'].replace('Epithelioid', 0.0)
    df_patient['Meso_type'] = df_patient['Meso_type'].replace('Sarcomatoid', 1.0)
    df_patient['Meso_type'] = df_patient['Meso_type'].replace('Biphasic', 1.0)

    # replace the nan values with string 'unknown'
    df_patient['wcc_score'] = df_patient['wcc_score'].replace(np.nan, 'Unknown')
    df_patient['desmoplastic_component'] = df_patient['desmoplastic_component'].replace(np.nan, 'Unknown')
    df_patient['smoking_history'] = df_patient['smoking_history'].replace(np.nan, 'Unknown')

    df_representations = pd.read_csv(representation_csv, index_col=0)
    # Extracting the case number from the samples
    df_representations['case_number'] = df_representations['samples'].apply(lambda x: int(x.split('_')[1]))
    #merge the two dataframes by case number
    df_representations = pd.merge(df_representations, df_patient, on='case_number', how='left')
    df_representations['samples'] = df_representations['samples'][:].astype(str)
    df_representations['slides'] = df_representations['slides'][:].astype(str)
    df_representations['tiles'] = df_representations['tiles'][:].astype(str)
    df_representations['hist_subtype'] = df_representations['hist_subtype'][:].astype(str)
    df_representations['type'] = df_representations['type'][:].astype(str)
    df_representations['stage'] = df_representations['stage'][:].astype(str)
    df_representations['smoking_history'] = df_representations['smoking_history'][:].astype(str)
    df_representations['wcc_score'] = df_representations['wcc_score'][:].astype(str)
    df_representations['desmoplastic_component'] = df_representations['desmoplastic_component'][:].astype(str)

    df_representations.to_csv(csv_results_path) 
    print(df_representations.head())
    print('csv file is created')
    return df_representations




def create_csv_tma(patient_csv, representation_csv, csv_results_path):
    #delete the csv file if it exists
    if os.path.exists(csv_results_path):
        os.remove(csv_results_path)
    
    # reading csv file and get the slides and samples and change the column name as wanted
    df_patient = pd.read_csv(patient_csv)
    df_patient = df_patient[['samples', 'slides', 'type']]
    df_representations = pd.read_csv(representation_csv, index_col=0)

    #merge the two dataframes by samples   
    df_representations = pd.merge(df_representations, df_patient, on='slides', how='inner')
    if df_representations['samples_x'].equals(df_representations['samples_y']):
        df_representations = df_representations.drop(columns=['samples_y'])
        df_representations = df_representations.rename(columns={'samples_x': 'samples'})
    else:
        print('Error in merging the two dataframes!')
        return

    df_representations['samples'] = df_representations['samples'][:].astype(str)
    df_representations['slides'] = df_representations['slides'][:].astype(str)
    df_representations['tiles'] = df_representations['tiles'][:].astype(str)
 
    df_representations.to_csv(csv_results_path)
    print('df_representations\n---------------------\n',df_representations)
    return df_representations





if __name__ == '__main__':
    csv_results_path = '%s/files/csv_lattice_tma_subsampled_he_complete_with_survival.csv' % main_path
    pkl_results_path = '%s/files/pkl_lattice_tma_subsampled_he_complete.pkl' % main_path
    h5_complete_path   = '%s/results/BarlowTwins_3/lattice_tma/h224_w224_n3_zdim128/hdf5_lattice_tma_he_complete.h5' % main_path

    representation_csv='%s/files/csv_lattice_tma_subsampled_he_complete.csv' % main_path
    # patient_csv='%s/files/patient_csv_fitered.csv' % main_path
    patient_csv = '%s/files/lattice_tma_map.csv' % main_path

    if not os.path.exists(representation_csv):
        print('Creating csv file from h5 file')
        df_representations = creat_csv_from_h5_tma(representation_csv, h5_complete_path)

    # add the survival information to the csv file
    # df_representations = create_csv_wsi(patient_csv=patient_csv\
    #                                 , representation_csv=representation_csv,\
    #                                     csv_results_path=csv_results_path)
    if not os.path.exists(csv_results_path):
        df_representations = create_csv_tma(patient_csv=patient_csv\
                                        , representation_csv=representation_csv,\
                                            csv_results_path=csv_results_path)
    else:
        df_representations = pd.read_csv(csv_results_path, index_col=0)

    # add sample slide column
    folds       = get_folds(df_representations, matching_field='slides', ind_column='samples', num_folds=5, valid_set=True)

    final_folds = fit_format(folds)

    # If no output, all good.
    sanity_check_overlap(folds, num_folds=5)

    store_data(final_folds, pkl_results_path)
    pkl_data = read_pkl(pkl_results_path)
