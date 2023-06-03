#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=10:00:00

source /home/${USER}/.bashrc
source activate rl

srun python3 $HOME/autometa/runs/run.py --algo=rl_squared --env=cheetah_velocity --prod
