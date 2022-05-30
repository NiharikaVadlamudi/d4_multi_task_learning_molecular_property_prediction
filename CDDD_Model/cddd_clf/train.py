'''
Training Process 

TorchMetrics : 
https://torchmetrics.readthedocs.io/en/stable/classification/confusion_matrix.html

'''

# Libraries
import csv
import sys
import csv 
import json
from random import shuffle
import numpy as np
import math

# python imports
import os
import sys 
import ast
import argparse
import warnings
import pandas as pd
import wandb
import functools

warnings.filterwarnings("ignore")

# torch imports
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics.functional import accuracy,precision,recall,auc,average_precision,confusion_matrix,auroc,spearman_corrcoef
from torch.utils.data import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

# File Imports 
from network import *
from dataclass import * 
from utils import *

# CUDA Device Settings 
print('---------- Device Information ---------------')
if torch.cuda.is_available():
    print('__CUDNN VERSION  : {} '.format(torch.backends.cudnn.version()))
    print('__Number CUDA Devices  : {}'.format(torch.cuda.device_count()))
    print('Allocated GPUs : {} , CPUs : {}'.format(torch.cuda.device_count(),os.cpu_count()))
    device= torch.device("cpu")
      
else:
    print('Allocated CPUs : {}'.format(os.cpu_count()))
    device=torch.device('cpu')
    print('Only CPU Allocation ..')
print('--------------------------------------------------')

taskList=['bioavail','cyp2d6','ames','herg']

# Parser 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name',default='reg_tdc_task',help='name of the project')
    parser.add_argument('--wid',default=wandb.util.generate_id(),help='wid initialisation')
    parser.add_argument('--expdir',default='tdc_reg',help='experiment folder')
    parser.add_argument('--mode',default='train',help='train/evaluate the model')
    parser.add_argument('--ntasks',default=1,type=int,help='N Classification Tasks')
    parser.add_argument('--learningRate',default=0.01,type=float,help='initial learning rate')
    parser.add_argument('--trainbatchSize',default=32,type=int,help='batchsize for train')
    parser.add_argument('--valbatchSize',default=256,type=int,help='batchsize for val')
    parser.add_argument('--maxEpochs',default=30,type=int,help='maximum epochs')
    parser.add_argument('--modelfile',default=None,help='Model Weight file path')
    parser.add_argument('--resume',default=None,type=int,help='Enter model checkpoint path')
    args = parser.parse_args()
    return args

# Folder Creation 
def create_folder(args):
    try : 
        path=args.expdir
        pathCheckpoints=args.expdir+'/checkpoints'
        pathResults=args.expdir+'/test/results/'
        if(args.mode=='train'):
            os.system('mkdir -p %s' % pathCheckpoints)
        if(args.mode=='test'):
            os.system('mkdir -p %s' % pathResults)
        print('Experiment folder created ')

        # WandB Initialisation 
        WANDB_API_KEY="4169d431880a3dd9883605d90d1fe23a248ea0c5"
        WANDB_ENTITY="amber1121"
        os.environ["WANDB_RESUME"] ="allow"
        os.environ["WANDB_RUN_ID"] = args.wid
        wandb.init(project="[Classification] D4 Molecular Property Prediction",id=args.wid,resume='allow',config=args)
        print('WandB login completed ..')

    except Exception as ex : 
        print('Folder Creation Error , Exiting : {}'.format(ex))
        sys.exit()


class Trainer(object):
    def __init__(self, args):
        self.args=args 
        # Training Parameters 
        self.trainbatchSize = args.trainbatchSize
        self.valbatchSize = args.valbatchSize
        self.ntasks=args.ntasks
        self.lr = args.learningRate
        self.maxEpochs=args.maxEpochs
        self.testCheckpointPath = args.modelfile
        self.expdir = args.expdir
        self.epoch=0

        # Create folders 
        create_folder(args)

        # Model Initialisation 
        self.model=CDDD_FC()
        self.model.to(device)

        # Specify the data csv files  
        
        train_df = pd.read_csv('./data/cddd_reg_task_aqsol_train.csv')
        valid_df = pd.read_csv('./data/cddd_reg_task_aqsol_test.csv')

        # Converting Dataframes to Dataclass 
        train_dataset = Dataclass(train_df)
        valid_dataset = Dataclass(valid_df)

        # DataLoaders 
        self.train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=self.trainbatchSize,shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=self.valbatchSize,shuffle=True)

        # Optimisers 
        self.optimizer =torch.optim.Adam(
            self.model.parameters(),
            lr=args.learningRate,
            amsgrad=False)
        
        self.lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        # Loss functions - Cross Entropy for the existing classes 
        self.clf_loss_fn = torch.nn.BCEWithLogitsLoss()

        # Resume 
        if args.resume is not None:
            self.resume(self.args.resume)

    def resume(self,epochNumber):
        print(' Model Training Resumed .. !')
        checkpointPath = os.path.join(self.args.expdir, 'checkpoints', 'epoch_%d.pth'%(epochNumber))
        save_state = torch.load(checkpointPath)
        try :
            self.model.load_state_dict(save_state['network_state_dict'])
            print('Model loaded ...  ')
        except Exception as ex :
            print('Model loading error ___ {}'.format(ex)) 
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        print('Model reloaded to resume from Epoch %d' % (self.epoch))
    
    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'network_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }
        save_name = os.path.join(self.args.expdir, 'checkpoints', 'epoch_%d.pth'%(epoch))
        torch.save(save_state, save_name)
        print('Saved model @ Epoch : {}'.format(self.epoch))
    

    def loop(self):
        for epoch in range(self.epoch,self.maxEpochs):
            self.epoch = epoch
            wandb.log({'current_lr':self.optimizer.param_groups[0]['lr'],'epoch':self.epoch})
            self.train(self.epoch)
            print('Epoch : {} ---- Instances Length : {} '.format(self.epoch,len(self.train_loader)))

    def train(self,epoch):
        print('Starting training...')
        self.model.to(device)
        self.model.train()
        countErr=0
        running_bce_loss = []
        print('Train Epoch ......{}'.format(epoch))
        for step,samples in enumerate(self.train_loader):
            try:
                self.optimizer.zero_grad()
                drug_graphs=samples[0]
                drug_graphs = torch.tensor(drug_graphs).to(device).float()
                taskMasks=torch.tensor((samples[1])).reshape((self.trainbatchSize,1,self.ntasks)).to(device).float()
                labels=torch.tensor(samples[2]).to(device).float()
                y_out = self.model([drug_graphs])

                y_out_mashed = y_out.reshape((1,self.trainbatchSize*self.ntasks))
                y_gd_mashed=labels.reshape((1,self.trainbatchSize*self.ntasks))
                taskMask_mashed = taskMasks.reshape((1,self.trainbatchSize*self.ntasks)).to(torch.bool)

                y_out_filtered = y_out_mashed[taskMask_mashed==True]
                y_gd_filtered = y_gd_mashed[taskMask_mashed==True]



                clf_loss = self.clf_loss_fn(y_out_filtered,y_gd_filtered)
                clf_loss.backward()

                self.optimizer.step()
                running_bce_loss.append(clf_loss.cpu().detach().numpy())
            
            except Exception as exp : 
                countErr=countErr+1
                print('TrainException : {} Errors : {}'.format(exp,countErr))
                continue
        
        # Apply validation after every epoch ..  
        metrics_dict,val_loss=self.validate(epoch)
        self.lr_decay.step(val_loss)

        print('\n')
        print(' Stats for epoch : {}'.format(epoch))
        print("TRAIN - Epoch: {}  Training loss : {} ".format(epoch,(np.mean(np.array(running_bce_loss)))))
        print("VALIDATE - Epoch: {} , Validation loss : {} ".format(epoch,val_loss))

        # WandB Logging 
        wandb.log({'avg_train_error':np.mean(np.array(running_bce_loss)),'epoch':epoch})
        wandb.log({'learning_rate':self.optimizer.param_groups[0]['lr'],'epoch':epoch})
        wandb.log({'train_data_errors':countErr,'epoch':epoch})
        wandb.log({'avg_validation_error':val_loss,'epoch':epoch})
        
        for i in range(self.ntasks):
            for key in metrics_dict['task_'+str(taskList[i])].keys():
                wandb.log({'task_{}_{}'.format(str(taskList[i]),key):metrics_dict['task_'+str(taskList[i])][key],'epoch':epoch})

        # Save model checkpoint ..
        self.save_checkpoint(self.epoch)

    def validate(self,epoch):
        # Set the model to evaluation model 
        errorPoints=0
        total_loss=0.0
        epoch_loss=[]
        tasks=[]
        # Metrics holders for all 
        metrics=['acc','auprc','auroc','spcoeff']
        for i in range(self.ntasks):
            task={'acc':[],'auprc':[],'auroc':[],'spcoeff':[]}
            tasks.append(task)
        
        # Iterating through samples .. 
        self.model.eval() 
        running_val_loss=[]
        print('Validation....')
        for step,samples in enumerate(self.valid_loader):
            try:
    
                drug_graphs=samples[0]
                drug_graphs = torch.tensor(drug_graphs).to(device).float()
                taskMasks=torch.tensor((samples[1])).reshape((self.valbatchSize,1,self.ntasks)).to(device).float()
                labels=torch.tensor(samples[2]).to(device).float()
                y_out = self.model([drug_graphs])
                y_out_mashed = y_out.reshape((1,self.valbatchSize*self.ntasks)).float()
                y_gd_mashed=labels.reshape((1,self.valbatchSize*self.ntasks)).float()
                taskMask_mashed = taskMasks.reshape((1,self.valbatchSize*self.ntasks)).to(torch.bool)
                y_out_filtered = y_out_mashed[taskMask_mashed==True]
                y_gd_filtered = y_gd_mashed[taskMask_mashed==True]

                # Compute metrics 
                clf_loss = self.clf_loss_fn(y_out_filtered,y_gd_filtered)
                running_val_loss.append(clf_loss.cpu().detach().numpy())

                # Metrics 
                y_out = y_out.reshape((self.valbatchSize,1,self.ntasks))

                for i,task in enumerate(tasks):
                    y_out_i=y_out[:,:,i]
                    y_gnd_i=labels[:,:,i]
                    task_mask_i=taskMasks[:,:,i]
                    # So, we have fixed on a task , we need to apply the metrics 
                    y_out_i =  y_out_i[task_mask_i==True]
                    y_gnd_i =  y_gnd_i[task_mask_i==True]
            
                    if(y_out_i.nelement() <=1 or  y_gnd_i.nelement() <=1):
                        continue 
                    else:
                        auprc_val = auprc(y_gnd_i,y_out_i)
                        spearman_val = spearman_corrcoef(y_gnd_i,y_out_i)
                        auroc_val = auroc(y_out_i,y_gnd_i.to(torch.int),pos_label=1)
                        acc_val = accuracy(y_out_i,y_gnd_i.to(torch.int))
                        if(not math.isnan(auprc_val)):
                            task['auprc'].append(auprc_val)
                            task['spcoeff'].append(spearman_val)
                            task['auroc'].append(auroc_val)
                            task['acc'].append(acc_val)
                            # print(i," ",task)
            except Exception as exp : 
                errorPoints+=1
                print('ValException : {}'.format(exp))
                continue 
                        
        # Collate everything .. 
        finalDict={}
        for i,task in enumerate(tasks):
            finalDict['task_'+str(taskList[i])]={}
            for metric in metrics :
                finalDict['task_'+str(taskList[i])][metric]=np.mean(np.asarray(task[metric],dtype=np.float))
        
        # Average MSE Loss
        epoch_clf_net_loss = np.mean(np.array(running_val_loss,dtype=np.float).flatten())
        
        # WandB logging 
        print('Associcated Loss : {}'.format(epoch_clf_net_loss))
        
      
        # WandB logging 
        wandb.log({'validation_data_errors':errorPoints,'epoch':epoch})
        return(finalDict,epoch_clf_net_loss)

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    if(args.mode=='train'):
        trainer.loop()
    else:
        trainer.test() 
  


