#!/bin/bash --login
#
#SBATCH --job-name=leiden_assignation
#SBATCH --mail-user=f.seyedshahi.1@research.gla.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/leiden_assignation_%j.out
#SBATCH --error=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/leiden_assignation_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --partition=compute-low-priority

export TZ=Europe/London
date
hostname

source ~/.bashrc
source "/mnt/cephfs/home/users/fshahi/miniconda3/etc/profile.d/conda.sh"
conda activate rapid


python3 ./run_representationsleiden_assignment.py \
    --meta_field 750K \
    --folds_pickle ./files/pkl_Meso_he_test_train_slides.pkl \
    --h5_complete_path ./results/BarlowTwins_3/Meso/h224_w224_n3_zdim128/hdf5_Meso_he_complete_filtered_metadata.h5 \
    --h5_additional_path ./results/BarlowTwins_3/MesoGraph/h224_w224_n3_zdim128/hdf5_MesoGraph_he_complete_new.h5


echo "Finished leiden clustering assignation"
date
exit 0