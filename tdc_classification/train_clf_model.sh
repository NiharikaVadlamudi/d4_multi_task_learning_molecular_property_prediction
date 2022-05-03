#!/bin/bash
#SBATCH --job-name="TDC Multi-Class Classification Training"
#SBATCH -A research
#SBATCH --gres=gpu:1
#SBATCH -n 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=./TDC/CLF/e0_train.txt
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END

python3 trainClf.py  --project_name 'tdc_clf_task' --expdir './TDC/CLF/' --mode 'train' --ntasks 4 --learningRate 0.05 --batchSize 1 --maxEpochs 1 
