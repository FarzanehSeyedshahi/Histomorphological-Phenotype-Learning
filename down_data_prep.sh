#!/bin/bash --login
#
#SBATCH --job-name=down_data_prep
#SBATCH --output=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/down_data_prep_%j.out
#SBATCH --error=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/down_data_prep_%j.err
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
conda activate HPL

python3 ./workflow/subtype_data_preparation.py --meta_folder 900 --dataset MesoGraph --transformation clr --pkl_path ./files/pkl_Meso_he_test_train_slides.pkl --additional_dataset MesoGraph

# python3 ./workflow/survival_data_preparation.py --meta_folder 750K --dataset Meso --transformation clr --pkl_path ./files/pkl_Meso_he_test_train_cases.pkl --additional_dataset TCGA_MESO

# python3 ./workflow/down_data_prep_additional.py --meta_folder 900 --dataset MesoGraph --transformation clr --pkl_path ./files/pkl_MesoGraph_he_test_train_slides.pkl --additional_dataset MesoGraph

date
exit 0