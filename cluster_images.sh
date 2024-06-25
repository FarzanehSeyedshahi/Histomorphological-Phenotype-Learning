#!/bin/bash --login
#
#SBATCH --job-name=cluster_images
#SBATCH --output=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/cluster_images_%j.out
#SBATCH --error=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/cluster_images_%j.err
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
conda activate rapid

python3 ./report_representationsleiden_samples.py --meta_folder 750K --meta_field 750K --matching_field slides --resolution 9.0 --fold 4 --h5_complete_path results/BarlowTwins_3/Meso/h224_w224_n3_zdim128/hdf5_Meso_he_complete_filtered_metadata.h5 --dpi 1000 --dataset Meso --tile_img --additional_as_fold

date
exit 0