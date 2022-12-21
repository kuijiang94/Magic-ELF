#!/bin/bash
#SBATCH --no-requeue
#SBATCH --account=zywang4
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH -o jk_ALformer.log
module load scl/gcc4.9
source activate torch1_1
python train_ALformer.py