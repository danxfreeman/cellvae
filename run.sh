#!/bin/bash
#SBATCH -J cnn
#SBATCH -t 0-12
#SBATCH -p short
#SBATCH --mem=64GB
#SBATCH -c 4

# Run model.
cd /home/daf179/cellvae/
/home/daf179/miniconda3/envs/cnn/bin/python experiment.py
