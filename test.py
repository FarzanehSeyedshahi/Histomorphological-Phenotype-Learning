# !pip install adjustText

from matplotlib        import collections             as matcoll
# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.colors import TwoSlopeNorm
# from matplotlib.pyplot import rc_context
from scipy.cluster     import hierarchy
# from adjustText        import adjust_text

import matplotlib.pyplot as plt
# import matplotlib
import seaborn as sns
import scanpy as sc
import pandas as pd
import numpy as np
import rapids_singlecell as rsc

# import math
# import glob
# import h5py
import sys
import os

main_path = '/mnt/cephfs/sharedscratch/users/fshahi/Projects/Histomorphological-Phenotype-Learning'
import warnings
warnings.filterwarnings("ignore")

sys.path.append(main_path)
# from models.clustering.logistic_regression_leiden_clusters import *
from models.evaluation.folds import load_existing_split
from models.clustering.correlations import *
from models.clustering.data_processing import *
# from models.clustering.leiden_representations import include_tile_connections_frame
# from data_manipulation.utils import store_data
from IPython.display import clear_output
clear_output()


# Resolution and fold for the tile clustering and slide representations.
resolution     = 2.0
fold_number    = 4
groupby        = 'leiden_%s' % resolution

# Folder run.
dataset     = 'Meso'
additional_dataset = 'TCGA_MESO'
meta_folder     = '750K'
# meta_folder     = 'meso_overal_survival_nn400'

matching_field  = 'slides'
# matching_field2 = 'samples'
# meta_field      = 'Meso_type'

# Penalties for Cox regression and flag for usage.
# use_cox        = False
# alpha          = 1.0

# Pickle files.
folds_pickle = '{}/files/pkl_{}_he_test_train_slides.pkl'.format(main_path, dataset)

# Tile representation files.
h5_complete_path   = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/hdf5_{}_he_complete_filtered_metadata.h5'.format(main_path, dataset, dataset)
h5_additional_path = '{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/hdf5_{}_he_complete_filtered_metadata.h5'.format(main_path, additional_dataset, additional_dataset)

# Run path.
main_cluster_path = h5_complete_path.split('hdf5_')[0]
main_cluster_path = os.path.join(main_cluster_path, meta_folder)
adatas_path       = os.path.join(main_cluster_path, 'adatas')
figure_path       = os.path.join(main_cluster_path, 'leiden_%s_fold%s' % (str(resolution).replace('.','p'),fold_number))
figure_path       = os.path.join(figure_path,       'figures')
if not os.path.isdir(figure_path):
    os.makedirs(figure_path)

# PAGA.

adata_train, h5ad_path = read_h5ad_reference(h5_complete_path, meta_folder, groupby, fold_number)

if os.path.isfile(h5ad_path.replace('.h5ad', '_paga.h5ad')):
    adata_train = anndata.read_h5ad(h5ad_path.replace('.h5ad', '_paga.h5ad'))
else:
    # sc.tl.paga(adata_train, groups=groupby, neighbors_key='nn_leiden')
    # rsc.tl.paga(adata_train, groups=groupby, neighbors_key='nn_leiden')
    # sc.pl.paga(adata_train, layout='rt_circular', random_state=0, threshold=0.29, single_component=True, node_size_scale=15, node_size_power=1,
            # edge_width_scale=0.2, fontsize=10, fontoutline=5, frameon=False, show=True)
    rsc.tl.umap(adata_train, neighbors_key='nn_leiden')
    # sc.tl.umap(adata_train, init_pos="paga", neighbors_key='nn_leiden')
    # sc.tl.tsne(adata_train, use_rep='X', perplexity=50, learning_rate=1000, random_state=42)
    adata_train.write(h5ad_path.replace('.h5ad', '_paga.h5ad'))