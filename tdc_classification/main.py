# Library Imports
import os
import sys
import csv
import json 
import argparse
import warnings
import numpy as np
import pandas as pd 
from copy import deepcopy
from collections import OrderedDict

# torch imports 
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


# File Imports 
from model import D4MP 
from dataclass import * 
from train import train,validate


# Parser 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='train',help='train/evaluate the model')
    parser.add_argument('--learningRate',default=0.001,help='Benchmark train / test split')
    parser.add_argument('--batchSize',default=32,help='batchsize for both test/train')
    parser.add_argument('--maxEpochs',default=50,help='maximum epochs')
    parser.add_argument('--testcheckpoint',default=None,help='Model Weight file path ,')
    args = parser.parse_args()
    return args


def main(args):

    # Specify the data csv files  
    train_df = pd.read_csv('./data/clf_train.csv')
    valid_df = pd.read_csv('./data/clf_test.csv')

    # converting dataframes to dataloaders
    train_dataset = Dataclass(train_df)
    valid_dataset = Dataclass(valid_df)

    # Done 
    train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=args.batchSize, shuffle=True)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=args.batchSize)
    
    # Done 
    model = D4MP()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate)
    scheduler = ReduceLROnPlateau(optimizer,patience=2,mode='min',verbose=True)

    if(args.mode=='train'):
        train(args.maxEpochs, model, optimizer, scheduler,train_loader,valid_loader,args.batchSize,ntasks=4,lambda1=1.0,lambda2=1.0,project_name='tdc_clf',thresh_epoch=5)
        print('Training Process is completed...')
    

if __name__ == '__main__':
    args = get_args()
    res=main(args)
    
   