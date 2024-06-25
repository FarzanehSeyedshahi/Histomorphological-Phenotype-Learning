#!/bin/bash --login
#
#SBATCH --job-name=leiden_assignation
#SBATCH --output=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/leiden_assignation_%j.out
#SBATCH --error=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/leiden_assignation_%j.err
#SBATCH --mail-user=2838247s@student.gla.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --cpus-per-task=10
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --partition=compute-low-priority

export TZ=Europe/London
date
hostname

source ~/.bashrc
source "/mnt/cephfs/home/users/fshahi/miniconda3/etc/profile.d/conda.sh"
# conda info --envs
conda activate HPL

# python3 ./run_representationsleiden_assignment.py --meta_field meso_subtypes_nn400 --folds_pickle ./files/pkl_Meso_400_subsampled_he_complete.pkl --h5_complete_path ./results/BarlowTwins_3/Meso_400_subsampled/h224_w224_n3_zdim128_filtered/hdf5_Meso_400_subsampled_he_complete_combined_metadata_filtered.h5 --h5_additional_path ./results/BarlowTwins_3/Meso_400_subsampled/h224_w224_n3_zdim128/hdf5_Meso_500_he_complete_metadata.h5

python3 ./run_representationsleiden_assignment.py \
    --resolution 5.0 \
    --meta_field removal \
    --folds_pickle ./files/pkl_Meso_400_subsampled_he_complete.pkl \
    --h5_complete_path ./results/BarlowTwins_3/Meso_400_subsampled/h224_w224_n3_zdim128/hdf5_Meso_400_subsampled_he_complete_combined_metadata.h5 \
    --h5_additional_path ./results/BarlowTwins_3/Meso/h224_w224_n3_zdim128/hdf5_Meso_he_complete.h5


echo "Finished leiden clustering assignation"
date
exit 0