#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=06:00:00

source /home/${USER}/.bashrc
source activate rl

srun python3 $HOME/autometa/runs/run.py --algo=auto_dr --env=ant_velocity --prod
