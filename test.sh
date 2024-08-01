#!/bin/bash --login
#
#SBATCH --job-name=umap
#SBATCH --mail-user=f.seyedshahi.1@research.gla.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/umap_%j.out
#SBATCH --error=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/umap_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --time=240:00:00
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=shard:40
#SBATCH --partition=gpu


export TZ=Europe/London

date
hostname
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
source "/mnt/cephfs/home/users/fshahi/miniconda3/etc/profile.d/conda.sh"
conda activate rapids_singlecell


python3 /nfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning/test.py 

date
exit 0