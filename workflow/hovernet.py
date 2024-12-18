import warnings
import json
import numpy as np
import os
import pandas as pd
import argparse
import submitit

warnings.filterwarnings("ignore")


def get_color_dict():
    color_dict = {
    "0" : ["nolabe", [0  ,   0,   0]], 
    "1" : ["neopla", [255,   0,   0]], 
    "2" : ["inflam", [0  , 255,   0]], 
    "3" : ["connec", [0  ,   0, 255]], 
    "4" : ["necros", [255, 255,   0]], 
    "5" : ["no-neo", [255, 165,   0]] }
    return color_dict



# def get_dict_from_json():
#     color_dict = get_color_dict()
#     path_folder = '/nfs/home/users/fshahi/Projects/Datasets/large_ndpis/tiled/'
#     filenames_5x = os.listdir(path_folder)

#     nuc_types = dict()
#     for filename in filenames_5x[:1]:
#         nuc_types[filename] = dict()
#         tiles = os.listdir(path_folder + filename + '/5.0/')
#         for tile in tiles[:1]:
#             start_i_5x, start_j_5x = tile.split('.jpeg')[0].split('_')
#             start_i, start_j = int(start_i_5x)*4, int(start_j_5x)*4
#             nuc_types[filename][tile] = dict()
#             for i in range(start_i, start_i+4):
#                 for j in range(start_j, start_j+4):
#                     try:
#                         path = '/nfs/home/users/fshahi/Projects/Datasets/large_ndpis/hovernet/{}/20.0/json/{}_{}.json'.format(filename,i,j)
#                         json_file = open(path)
#                         data = json.load(json_file)
#                         for i_nuc in data['nuc']:
#                             type_index = str(data['nuc'][i_nuc]['type'])
#                             nuc_type_key = color_dict[type_index][0]
#                             nuc_types[filename][tile][nuc_type_key] = nuc_types[filename][tile].get(nuc_type_key, 0) + 1
#                     except:
#                         pass


def csv_from_json(csv_path, path_folder):
    color_dict = get_color_dict()
    filenames_5x = os.listdir(path_folder)

    nuc_types_df = pd.DataFrame(columns=['slides', 'tiles', 'nolabe', 'neopla', 'inflam', 'connec', 'necros', 'no-neo'])
    for filename in filenames_5x:
        try:
            tiles = os.listdir(path_folder + filename + '/5.0/')
            for tile in tiles:
                i_5x, j_5x = tile.split('.jpeg')[0].split('_')
                i_5x, j_5x = int(i_5x), int(j_5x)
                start_i, start_j = i_5x*4, j_5x*4
                row = pd.DataFrame([[filename, tile, 0, 0, 0, 0, 0, 0]], columns=['slides', 'tiles', 'nolabe', 'neopla', 'inflam', 'connec', 'necros', 'no-neo'])
                for i in range(start_i, start_i+4):
                    for j in range(start_j, start_j+4):
                        try:
                            path = '/nfs/home/users/fshahi/Projects/Datasets/large_ndpis/hovernet/{}/20.0/json/{}_{}.json'.format(filename,i,j)
                            json_file = open(path)
                            data = json.load(json_file)
                            for i_nuc in data['nuc']:
                                type_index = str(data['nuc'][i_nuc]['type'])
                                nuc_type_key = color_dict[type_index][0]
                                row[nuc_type_key] = row[nuc_type_key] + 1
                        except:
                            continue
                nuc_types_df = pd.concat([nuc_types_df, row], axis=0)
            nuc_types_df.to_csv(csv_path, index=False, mode='a', header=False)
        except:
            continue
    return nuc_types_df


def merge_df_clusters(nuc_types_df, cluster_csv_path, path_folder):
    filenames_5x = os.listdir(path_folder)   
    # fetch clusters
    clusters = pd.read_csv(cluster_csv_path)
    clusters = clusters[['tiles', 'leiden_2.0', 'slides']]
    clusters['slides'] = clusters['slides'].apply(lambda x: x+'_files') # add _files to the slides to match dictionary
    clusters = clusters[clusters['slides'].isin(filenames_5x)] # filter out the slides that are in the dictionary


    # merge the dataframes
    df = nuc_types_df.merge(clusters, on=['slides', 'tiles'], how='inner')
    print('Number of slides:',df['slides'].value_counts().shape[0])
    df_path = cluster_csv_path.split('adatas')[0] + 'nuc_types_hovernet_v2.csv'
    df.to_csv(df_path, index=False)

    plotting_dfs = df.drop('slides', axis=1)
    plotting_dfs = plotting_dfs.groupby('leiden_2.0')
    tiles_per_cluster = [len(plotting_dfs.get_group(i)) for i in plotting_dfs.groups]
    plotting_dfs = plotting_dfs.sum()
    plotting_dfs['tiles'] = tiles_per_cluster
    plotting_dfs['mean_inflammation'] = plotting_dfs['inflammation']/plotting_dfs['tiles']
    plotting_dfs['mean_necrosis'] = plotting_dfs['necrosis']/plotting_dfs['tiles']
    plotting_dfs['mean_connective'] = plotting_dfs['connective']/plotting_dfs['tiles']
    plotting_dfs = plotting_dfs.reset_index()
    plotting_dfs.to_csv(cluster_csv_path.split('adatas')[0] + 'nuc_types_hovernet_clusters_v2.csv', index=False)



def main():
    csv_path = '/nfs/home/users/fshahi/Projects/Datasets/large_ndpis/tiled/nuc_types_full_v2.csv'
    path_folder = '/nfs/home/users/fshahi/Projects/Datasets/large_ndpis/tiled/'
    cluster_csv_path = '/nfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/Meso/h224_w224_n3_zdim128/750K/adatas/Meso_he_complete_filtered_metadata_leiden_2p0__fold4_v2.csv'
    nuc_types_df = csv_from_json(csv_path, path_folder)
    # nuc_types_df = pd.read_csv(csv_path)
    # drop the replicates
    nuc_types_df = nuc_types_df.drop_duplicates()
    nuc_types_df.to_csv(csv_path.replace('.csv', 'cleaned.csv'), index=False)
    nuc_types_df.columns = ['slides', 'tiles', 'necrosis', 'neoplastic', 'inflammation', 'connective', 'no-neo', 'nolabe']
    merge_df_clusters(nuc_types_df, cluster_csv_path, path_folder)


def launch():
    executor = submitit.AutoExecutor(
        folder=os.path.join(args.folder, 'job_%j'))
    executor.update_parameters(
        slurm_partition=args.partition,
        timeout_min=args.time,
        nodes=args.nodes,
        # gpus_per_node=args.gpu_per_node,
        slurm_mem = args.slurm_mem,
        # cpus_per_task=args.num_cpus,
        name=args.job_name,
        )
    print('executor params:', executor.parameters)


    with executor.batch():
        # job = executor.submit(csv_from_json)
        job = executor.submit(main)

	
    print(job.job_id)
    # print(job.result())


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='slurm_logs')
    parser.add_argument('--partition', type=str, default='compute-low-priority')
    parser.add_argument('--time', type=int, default=4300)
    parser.add_argument('--nodes', type=int, default=1)
    # parser.add_argument('--gpu_per_node', type=int, default=1 )
    parser.add_argument('--slurm_mem', type=str, default='200G')
    # parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--job_name', type=str, default='hovernetjson')

    args = parser.parse_args()
    launch()
