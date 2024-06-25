#!/bin/bash --login
#
#SBATCH --job-name=filtering
#SBATCH --output=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/filtering_%j.out
#SBATCH --error=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/filtering_%j.err
#SBATCH --mail-user=2838247s@student.gla.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --partition=compute-low-priority

export TZ=Europe/London
date
hostname

source ~/.bashrc
source "/mnt/cephfs/home/users/fshahi/miniconda3/etc/profile.d/conda.sh"
# conda info --envs
conda activate HPL

python3 utilities/tile_cleaning/remove_indexes_h5.py --pickle_file files/indexes_to_remove/Meso_400_subsampled/Meso.pkl --h5_file results/BarlowTwins_3/Meso/h224_w224_n3_zdim128/hdf5_Meso_he_complete.h5

date
exit 0