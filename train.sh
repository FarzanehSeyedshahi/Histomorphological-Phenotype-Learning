#!/bin/bash --login
#
#SBATCH --job-name=train
#SBATCH --mail-user=f.seyedshahi.1@research.gla.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/train_%j.out
#SBATCH --error=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/train_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --time=240:00:00
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=shard:40
#SBATCH --partition=gpu


export TZ=Europe/London

date
hostname

source "/mnt/cephfs/home/users/fshahi/miniconda3/etc/profile.d/conda.sh"
conda activate HPL


python3 run_representationspathology.py \
--img_size 224 \
--batch_size 64 \
--epochs 60 \
--z_dim 128 \
--model BarlowTwins_3 \
--dataset Meso \
--check_every 10 \
--report \
--restore

date
exit 0