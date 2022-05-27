#!/bin/bash
#SBATCH --job-name="CDDD Classification : TDC Multi-Class Clf Training"
#SBATCH -A research
#SBATCH -n 10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=./TDC/cddd_clf_0/exp0_train.txt
#SBATCH --mail-user=niharika.vadlamudi@research.iiit.ac.in
#SBATCH --mail-type=END
#SBATCH --job-name=1_cddd_clf_lr_0.001
#SBATCH --gres=gpu:1

python3 train.py --wid 'cddd_e0_clf_tdc' --project_name 'tdc_clf_task_v0' --expdir './TDC/cddd_clf_0/' --mode 'train' --ntasks 4 --learningRate 0.001 --trainbatchSize 64 --valbatchSize 256 --maxEpochs 30
