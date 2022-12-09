#!/bin/bash

#SBATCH --job-name=AaD_ALL_BN
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=160000
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mx648@nyu.edu
#SBATCH --output=AaD_ALL_BN_%j.out
#SBATCH --error=AaD_ALL_BN_error_%j.out
#SBATCH -p aquila
#SBATCH --gres=gpu:2
#SBATCH --constraint=2080Ti

module purge
module load cuda/10.2
source ~/.bashrc
cd /scratch/mx648/Capstone/Capstone_Project_Fall_2022/AaD_SFDA-main
source activate mycapstone
echo "start training"
python tar_adaptation_BN_all.py>AaD_ALL_BN_Repeat.log
echo "Ending"
