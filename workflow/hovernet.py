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
        "5" : ["no-neo", [255, 165,   0]] 
    }
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


def csv_from_json():
    color_dict = get_color_dict()
    path_folder = '/nfs/home/users/fshahi/Projects/Datasets/large_ndpis/tiled/'
    filenames_5x = os.listdir(path_folder)

    nuc_types_df = pd.DataFrame(columns=['slides', 'tiles', 'necrosis', 'neoplastic', 'inflammation', 'connective', 'no-neo', 'nolabe'])
    for filename in filenames_5x:
        tiles = os.listdir(path_folder + filename + '/5.0/')
        for tile in tiles:
            start_i_5x, start_j_5x = tile.split('.jpeg')[0].split('_')
            start_i, start_j = int(start_j_5x)*4, int(start_i_5x)*4
            row = pd.DataFrame([[filename, tile, 0, 0, 0, 0, 0, 0]], columns=['slides', 'tiles', 'necrosis', 'neoplastic', 'inflammation', 'connective', 'no-neo', 'nolabe'])
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
                        pass
            nuc_types_df = pd.concat([nuc_types_df, row], axis=0)
    nuc_types_df.to_csv('/nfs/home/users/fshahi/Projects/Datasets/large_ndpis/tiled/nuc_types.csv', index=False)
    return nuc_types_df

# def main():
#     nuc_types_df = csv_from_json()
#     nuc_types_df.to_csv('/nfs/home/users/fshahi/Projects/Datasets/large_ndpis/tiled/nuc_types.csv', index=False)


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
        job = executor.submit(csv_from_json)
	
    print(job.job_id)
    # print(job.result())


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='slurm_logs')
    parser.add_argument('--partition', type=str, default='compute-low-priority')
    parser.add_argument('--time', type=int, default=4300)
    parser.add_argument('--nodes', type=int, default=1)
    # parser.add_argument('--gpu_per_node', type=int, default=1 )
    parser.add_argument('--slurm_mem', type=str, default='128G')
    # parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--job_name', type=str, default='hovernetjson')

    args = parser.parse_args()
    launch()
