import h5py
# import scanpy as sc
import numpy as np
import pandas as pd

main_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning'
h5_complete_path   = '%s/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/hdf5_Meso_250_subsampled_he_complete.h5' % main_path
csv_complete_path   = '%s/csv_Meso_250_subsampled_he_complete.csv' % main_path


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