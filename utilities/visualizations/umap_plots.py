import warnings
warnings.filterwarnings('ignore')
from anndata.experimental.pytorch import AnnLoader

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors       import LinearSegmentedColormap
from skimage.transform       import resize
from plottify                import autosize
from sklearn                 import metrics
from PIL                     import Image
from adjustText              import adjust_text
from scipy.cluster           import hierarchy
import statsmodels.api   as sm
import matplotlib
import anndata
import random
import fastcluster
import copy
import umap
import h5py
import sys
import os
#filter warnings


import sys
main_path = '/raid/users/farzaneh/Histomorphological-Phenotype-Learning'
sys.path.append(main_path)
from models.clustering.correlations import *
from models.clustering.data_processing import *
from models.visualization.attention_maps import *
from data_manipulation.data import Data


# Read the CSV file
# # csv_filename = '{}Meso_250_subsampled_he_complete_combined_metadata_filtered_leiden_1p0__fold0.csv'.format(adatas_path)
# csv_filename = '{}Meso_250_subsampled_he_complete_combined_metadata_filtered_leiden_2p0__fold0.csv'.format(adatas_path)
# df = pd.read_csv(csv_filename)

# # Set up the figure with three separate axes
# fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# # Plot the number of each type
# axs[0].bar(df['Meso_type'].value_counts().index, df['Meso_type'].value_counts().values)
# #put the number of each type on top of the bar
# for i, v in enumerate(df['Meso_type'].value_counts().values):
#     axs[0].text(i - 0.1, v + 10, str(v))

# axs[0].set_title('Number of Each Type')

# # Plot the number of each stage
# axs[1].bar(df['stage'].value_counts().index, df['stage'].value_counts().values)
# axs[1].set_title('Number of Each Stage')

# # Plot the number of each sample filtered by leiden_1 column
# # axs[2].bar(df.loc[df['leiden_1.0'] == 1, 'samples'].value_counts().index,
# #             df.loc[df['leiden_1.0'] == 1, 'samples'].value_counts().values)
# # axs[2].set_title('Number of Each Sample (filtered by leiden_1)')

# axs[2].bar(df['leiden_2.0'].value_counts().index, df['leiden_2.0'].value_counts().values)
# axs[2].set_title('Number of Each Sample (leiden_2)')


# # Adjust the spacing between subplots
# fig.tight_layout()

# # Save the figure as a PNG file with the same name as the CSV file
# png_filename = csv_filename.replace('.csv', '.png')
# plt.savefig(png_filename)
   



# Representations and Cluster Network.
def show_umap_leiden(adata, meta_field, layout, random_state, threshold, node_size_scale, node_size_power, edge_width_scale, directory, file_name,
                     fontsize=10, fontoutline=2, marker_size=2, ax_size=16, l_size=12, l_t_size=14, l_box_w=1, l_markerscale=1, palette='tab20', figsize=(30,10),
                     leiden_name=False):


    leiden_clusters = np.unique(adata.obs[groupby].astype(int))
    colors = sns.color_palette(palette, len(leiden_clusters))

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 3, 1)
    print(meta_field)

    ax = sc.pl.umap(adata, ax=ax, color=meta_field, size=marker_size, show=False, frameon=False, na_color='black')
    if meta_field == 'Meso_type':
        legend_c = ax.legend(loc='best', markerscale=l_markerscale, title=meta_field, prop={'size': l_size})
        legend_c.get_title().set_fontsize(l_t_size)
        legend_c.get_frame().set_linewidth(l_box_w)
        legend_c.get_texts()[0].set_text('Epithelioid(0)')
        legend_c.get_texts()[1].set_text('Sarcomatoid(1)')
    ax.set_title('Tile Vector\nRepresentations', fontsize=ax_size, fontweight='bold')

    ax  = fig.add_subplot(1, 3, 2)
    sc.pl.umap(adata, ax=ax, color=groupby, size=marker_size, show=False, legend_loc='on data', legend_fontsize=fontsize, legend_fontoutline=fontoutline, frameon=False, palette=colors)
    if leiden_name:
        ax.set_title('Leiden Clusters', fontsize=ax_size, fontweight='bold')
    else:
        ax.set_title('Histomorphological Phenotype\nClusters', fontsize=ax_size, fontweight='bold')

    adjust_text(ax.texts)

    ax  = fig.add_subplot(1, 3, 3)
    names_lines = ['Epithelioid', 'Sarcomatoid']
    sc.pl.paga(adata, layout=layout, random_state=random_state, color=meta_field, threshold=threshold, node_size_scale=node_size_scale, node_size_power=node_size_power, edge_width_scale=edge_width_scale, fontsize=fontsize, fontoutline=fontoutline, frameon=False, show=False, ax=ax)
    if meta_field == 'Meso_type':
        legend = ax.legend(legend_c.legendHandles, names_lines, title=meta_field, loc='upper left', prop={'size': l_size})
        legend.get_title().set_fontsize(l_t_size)
        legend.get_frame().set_linewidth(l_box_w)
    if leiden_name:
        ax.set_title('Leiden Cluster Network', fontsize=ax_size, fontweight='bold')
    else:
        ax.set_title('Histomorphological Phenotype\nCluster Network', fontsize=ax_size, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(directory,file_name))
    plt.show()


def create_paga_umap_init(h5ad_path):
    done = False
    if os.path.isfile(h5ad_path.replace('.h5ad', '_paga.h5ad')):
        done=True
        adata_train = anndata.read_h5ad(h5ad_path.replace('.h5ad', '_paga.h5ad'))
    else:
        sc.tl.paga(adata_train, groups=groupby, neighbors_key='nn_leiden')
        sc.tl.umap(adata_train, init_pos="paga", neighbors_key='nn_leiden')
        adata_train.write(h5ad_path.replace('.h5ad', '_paga.h5ad'))
        done = True
    return done, adata_train

if __name__ == '__main__':
    '''
    obs: 'Meso_type', 'case_number', 'hist_subtype', 'indexes', 'labels', 'original_set', 'os_event_data', 'os_event_ind', 'patterns', 'samples', 'slides', 'stage', 'tiles', 'type', 'leiden_2.0'
    uns: 'leiden', 'nn_leiden', 'pca'
    obsm: 'X_pca'
    varm: 'PCs'
    obsp: 'nn_leiden_connectivities', 'nn_leiden_distances'
    '''
    
    h5ad_path = '%s/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/meso_nn250/adatas/Meso_250_subsampled_he_complete_combined_metadata_filtered_leiden_2p0__fold0_subsample.h5ad'%main_path
    adata = sc.read(h5ad_path)

    ############# mesosubtype
    # meta_field     = 'os_event_ind'
    meta_field = 'Meso_type'
    matching_field = 'slides'
    resolution     = 2.0
    fold_number    = 1
    groupby        = 'leiden_%s' % resolution
    meta_folder    = 'meso_nn250'
    folds_pickle   = '%s/files/pkl_Meso_250_subsampled_he_complete.pkl'%main_path
    additional_as_fold = False
    ############# 

    ############# addresses
    h5_complete_path = '%s/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/hdf5_Meso_250_subsampled_he_complete_combined_metadata_filtered.h5'%main_path
    # adatas_path = '%s/results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/meso_nn250/adatas/'%main_path
    file_name = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), fold_number)
    # Setup folder esquema
    main_cluster_path = h5_complete_path.split('hdf5_')[0]
    main_cluster_path = os.path.join(main_cluster_path, meta_folder)
    figures_path      = os.path.join(main_cluster_path, 'figures')
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
    ############# 

    ############# Graph visualization related
    layout           = 'fa'  # ‘fa’, ‘fr’, ‘rt’, ‘rt_circular’, ‘drl’, ‘eq_tree’
    random_state     = 0
    threshold        = 0.29

    # Figure related
    node_size_scale  = 25
    node_size_power  = 0.5
    edge_width_scale = .05
    fontsize    = 10
    fontoutline = 2
    meta_field   = 'Meso_type'
    ############# 


    # adata_train, h5ad_path = read_h5ad_reference(h5_complete_path, meta_folder, groupby, fold_number)

    done, adata_train = create_paga_umap_init(h5ad_path)
    if done:
        print('PAGA and UMAP initialized:', adata_train)

    sns.set_theme(style='white')
    show_umap_leiden(adata_train, meta_field, layout, random_state, threshold, node_size_scale, node_size_power, edge_width_scale, directory=figures_path,
                    file_name=file_name + '_clusternetwork_all_anno.jpg', fontsize=25, fontoutline=10, marker_size=5, ax_size=62, l_size=50, l_t_size=55, l_box_w=4,
                    l_markerscale=6, palette='tab20', figsize=(50,20))