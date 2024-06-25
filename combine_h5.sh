#!/bin/bash --login
#SBATCH --job-name=combine_h5
#SBATCH --output=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/combine_h5_%j.out
#SBATCH --error=/mnt/cephfs/home/users/fshahi/Projects/slurm_logs/combine_h5_%j.err
#SBATCH --mail-user=2838247s@student.gla.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=45G
#SBATCH --partition=interactive

export TZ=Europe/London
date
hostname

source ~/.bashrc
source "/mnt/cephfs/home/users/fshahi/miniconda3/etc/profile.d/conda.sh"
conda activate HPL

python3 ./utilities/h5_handling/combine_complete_h5.py --model BarlowTwins_3 --dataset MesoGraph

echo "Finished combining h5 files"
date
exit 0