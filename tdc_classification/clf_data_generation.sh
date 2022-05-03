#!/bin/bash

python3 data_preperation.py --tasktype 'clf'  --mode 'train' --fileName 'clf_train'
python3 data_preperation.py  --tasktype 'clf' --mode 'test' --fileName 'clf_test'

