#!/bin/bash --login
#
#SBATCH --job-name=filtering
#SBATCH --output=/mnt/cephfs/home/users/fshahi/sharedscratch/hpl_scripts/log-files/231122_Meso_500_clustering_%A.out
#SBATCH --nodes=1
#SBATCH --mail-user=2838247s@student.gla.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --partition=compute-low-priority

export TZ=Europe/London
date
hostname

source ~/.bashrc
source "/mnt/cephfs/home/users/fshahi/miniconda3/etc/profile.d/conda.sh"
# conda info --envs
conda activate HPL

python3 ./utilities/tile_cleaning/remove_indexes_h5.py --pickle_file ./files/indexes_to_remove/Meso_400_subsampled/Meso_500.pkl  --h5_file ./results/BarlowTwins_3/Meso_500/h224_w224_n3_zdim128/hdf5_Meso_500_he_complete_metadata.h5 --override

date
exit 0