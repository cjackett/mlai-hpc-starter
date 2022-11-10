#!/bin/bash -l

# Slurm submit script
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512m
#SBATCH --time=0-00:10:00

# Activate virtual environment
source env/bin/activate

# Train the model
srun python3 src/main.py
