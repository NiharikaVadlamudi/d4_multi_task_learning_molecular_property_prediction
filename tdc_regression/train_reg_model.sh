#!/bin/bash
#SBATCH --job-name="TDC Multi-Class Regression Training"
#SBATCH -A research
#SBATCH --gres=gpu:1
#SBATCH -n 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=./TDC/REG/e0_train.txt
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END

python3 trainReg.py  --project_name 'tdc_reg_task' --expdir './TDC/REG/' --mode 'train' --ntasks 6 --learningRate 0.05 --batchSize 1 --maxEpochs 1 
