#!/bin/bash

python3 data_preperation.py --tasktype 'reg'  --mode 'train' --fileName 'reg_train'
python3 data_preperation.py  --tasktype 'reg' --mode 'test' --fileName 'reg_test'

