#!/bin/bash --login
#SBATCH --job-name=metadata
#SBATCH --output=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/metadata_%j.out
#SBATCH --error=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/metadata_%j.err
#SBATCH --mail-user=2838247s@student.gla.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --partition=compute-low-priority

export TZ=Europe/London
date
hostname

source ~/.bashrc
source "/mnt/cephfs/home/users/fshahi/miniconda3/etc/profile.d/conda.sh"
conda activate HPL

###### Don't put the matching field in the list_meta_field | care about the override flag
# python3 /mnt/cephfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning/utilities/h5_handling/create_metadata_h5.py \
#  --meta_file /mnt/cephfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning/files/Meso_patients.csv \
#  --matching_field case_Id \
#  --list_meta_field type N_stage T_Stage TNM_Stage os_event_ind os_event_data smoking_history wcc_score desmoplastic_component Meso_type age Sex recurrence time_to_recurrence confident_diagnosis HB_score haemoglobin side diaphragm_involvement rib_involvement lung_involvement chest_wall_involvement \
#  --h5_file /mnt/cephfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/Meso/h224_w224_n3_zdim128/hdf5_Meso_he_complete_filtered.h5 \
#  --meta_name metadata

python3 /mnt/cephfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning/utilities/h5_handling/create_metadata_h5.py \
 --meta_file /mnt/cephfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning/MesoGraph/dataset/AIME_2022_Mesothelioma_Cores/labels.csv \
 --matching_field samples \
 --list_meta_field Meso_type \
 --h5_file /mnt/cephfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/MesoGraph/h224_w224_n3_zdim128/hdf5_MesoGraph_he_complete.h5 \
 --meta_name metadata




date
exit 0