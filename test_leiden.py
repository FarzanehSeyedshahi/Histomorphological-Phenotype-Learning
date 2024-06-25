from models.clustering.leiden_representations import *
from models.clustering.data_processing import *
from models.visualization.attention_maps import get_x_y
import warnings
warnings.filterwarnings('ignore')


h5_complete_path = '/mnt/cephfs/sharedscratch/users/fshahi/Projects/MIF/results/vit_small/Meso_500/h224_w224_n3_zdim384/hdf5_Meso_500_he_complete.h5'
meta_field       = 'Meso_500_he_dino'
rep_key          = 'z_latent'
resolutions      = [2.5]
n_neighbors      = 250
subsample        = 200000
include_connections = False

complete_frame, complete_dims, complete_rest = representations_to_frame(h5_complete_path, meta_field=meta_field, rep_key=rep_key)
train_frame = complete_frame[complete_frame['original_set'] == 'train']

# Setup folder scheme
main_cluster_path = h5_complete_path.split('hdf5_')[0]
main_cluster_path = os.path.join(main_cluster_path, meta_field)
main_cluster_path = os.path.join(main_cluster_path, 'adatas')

if not os.path.isdir(main_cluster_path):
    os.makedirs(main_cluster_path)

for resolution in resolutions:
    print('Leiden %s' % resolution)
    groupby = 'leiden_%s' % resolution
    adata_name = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s' % (groupby.replace('.', 'p'))
    adata_train, subsample = run_clustering(train_frame, complete_dims, complete_rest, resolution, groupby, n_neighbors, main_cluster_path, '%s_subsample' % adata_name,
                                                subsample=subsample, include_connections=include_connections, save_adata=True)
    








# python3 ./run_representationsleiden_assignment.py --resolution 2.0 --meta_field meso_subtypes_nn400 --folds_pickle ./files/pkl_Meso_400_subsampled_he_complete.pkl --h5_complete_path ./results/BarlowTwins_3/Meso_400_subsampled/h224_w224_n3_zdim128_filtered/hdf5_Meso_400_subsampled_he_complete_combined_metadata_filtered.h5 --h5_additional_path ./results/BarlowTwins_3/Meso_400_subsampled/h224_w224_n3_zdim128_filtered/hdf5_Meso_500_he_complete_metadata_filtered.h5