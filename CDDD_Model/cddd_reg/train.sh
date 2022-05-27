#!/bin/bash
#SBATCH --job-name="CDDD Regression : TDC Multi-Class Regression Training"
#SBATCH -A research
#SBATCH -n 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=./TDC/cddd_reg_1/exp0_train.txt
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END

python3 train.py  --resume 6  --wid 're_cddd_e0_reg_tdc' --project_name 'tdc_reg_task_v0' --expdir './TDC/cddd_reg_1/' --mode 'train' --ntasks 6 --learningRate 0.001 --batchSize 64  --maxEpochs 25 
