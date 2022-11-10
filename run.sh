#!/bin/bash -l

# Slurm submit script
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=2g
#SBATCH --time=0-00:10:00

# Activate virtual environment
source env/bin/activate

# Train the model
srun python3 src/main.py
