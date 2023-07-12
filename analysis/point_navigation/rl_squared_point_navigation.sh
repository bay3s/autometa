#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=06:00:00

source /home/${USER}/.bashrc
source activate rl

srun python3 $HOME/autometa/analysis/point_navigation/run_evaluation.py --algo=rl_squared --grid-size=3.0
