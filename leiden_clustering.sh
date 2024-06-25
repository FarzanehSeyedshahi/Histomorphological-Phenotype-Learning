#!/bin/bash --login
#
#SBATCH --job-name=leiden
#SBATCH --mail-user=f.seyedshahi.1@research.gla.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/leiden_%j.out
#SBATCH --error=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/leiden_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=28G
#SBATCH --partition=gpu-ke


export TZ=Europe/London

date
hostname

source "/mnt/cephfs/home/users/fshahi/miniconda3/etc/profile.d/conda.sh"
# conda info --envs
conda activate rapids_singlecell
# conda activate HPL
# conda activate rapid


python3 ./run_representationsleiden.py \
--meta_field test \
--matching_field slides \
--folds_pickle ./files/pkl_Meso_he_test_train_slides.pkl \
--h5_complete_path ./results/BarlowTwins_3/Meso/h224_w224_n3_zdim128/hdf5_Meso_he_complete_metadata.h5 \
--subsample 750000

date
exit 0