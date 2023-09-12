#!/bin/bash

#SBATCH --account=def-lcollins
#SBATCH --gpus-per-node=1          # Number of GPU(s) per node
#SBATCH --cpus-per-task=48         # CPU cores/threads
#SBATCH --mem=96000M               # memory per node
#SBATCH --time=3-12:00
#SBATCH --mail-user=reza.rajabli@mail.mcgill.ca
#SBATCH --mail-type=ALL

source ~/rajab-py3.8.10-env/bin/activate
cd ~/scratch/Rajab-SFCN-VAE-Regularized-CC-OS
nvidia-smi
srun python main.py
