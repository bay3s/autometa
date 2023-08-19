#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00

#source /home/${USER}/.bashrc
#source activate rl

for seed in 2878 5069 5073 5250 4420
  do
    python3 $HOME/autometa/analysis/point_navigation/run_evaluation.py --algo=rl_squared --grid-size=5.0 --random-seed=$seed
  done
