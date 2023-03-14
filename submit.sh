#!/bin/sh
#SBATCH --job-name=cellvae
#SBATCH --partition=medium
#SBATCH --time=00-24:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# Run VAE.
source activate cellvae
python run.py $1
