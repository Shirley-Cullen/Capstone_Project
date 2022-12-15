#!/bin/bash

#SBATCH --job-name=DUA-norm-wrn
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=160000
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mx648@nyu.edu
#SBATCH --output=DUA_%j.out
#SBATCH --error=DUA_error_%j.out
#SBATCH -p aquila
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080Ti

module purge
module load cuda/10.2
cd /scratch/mx648/Capstone/Capstone_Project_Fall_2022/DUA-master
source activate mycapstone
echo "start training"
python dua_without_BN.py>myTraining2.log --model wrn --dataroot Data --ckpt_path WRN.pt 
echo "Ending"