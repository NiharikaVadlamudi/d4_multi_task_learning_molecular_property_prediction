#!/bin/bash
#SBATCH --job-name="[AqSol] Single Task CDDD Regression Training"
#SBATCH -A research
#SBATCH -n 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=./single_cddd_reg_aqsol/exp0_train.txt
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END

python3 train.py  --wid 'aqsol_cddd_reg_tdc' --project_name 'single_task_cddd' --expdir './single_cddd_reg_aqsol/' --mode 'train' --ntasks 1 --learningRate 0.001 --batchSize 64  --maxEpochs 25 
