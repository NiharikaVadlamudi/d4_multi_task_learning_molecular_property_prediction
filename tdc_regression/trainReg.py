# Libraries
import os 
import sys
import csv 
import json
from random import shuffle
import numpy as np
from tqdm import tqdm

# python imports
import os
import sys 
import ast
import argparse
import warnings
import pandas as pd
import wandb
warnings.filterwarnings("ignore")

# torch imports
import torch
import torch.optim as optim
from torchmetrics.functional import accuracy,precision,recall,auc,mean_squared_error
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# File Imports 
from model import D4MP 
from dataclass import * 


# CUDA Device Settings 
print('---------- Device Information ---------------')
if torch.cuda.is_available():
    print('__CUDNN VERSION  : {} '.format(torch.backends.cudnn.version()))
    print('__Number CUDA Devices  : {}'.format(torch.cuda.device_count()))
    print('Allocated GPUs : {} , CPUs : {}'.format(torch.cuda.device_count(),os.cpu_count()))
    device= torch.device("cuda:0")
      
else:
    print('Allocated CPUs : {}'.format(os.cpu_count()))
    device=torch.device('cpu')
    print('Only CPU Allocation ..')
print('--------------------------------------------------')



# Parser 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name',default='reg_tdc_task',help='name of the project')
    parser.add_argument('--wid',default=wandb.util.generate_id(),help='wid initialisatio')
    parser.add_argument('--expdir',default='tdc_clf',help='experiment folder')
    parser.add_argument('--mode',default='train',help='train/evaluate the model')
    parser.add_argument('--ntasks',default=4,type=int,help='N Classification Tasks')
    parser.add_argument('--learningRate',default=0.01,type=float,help='initial learning rate')
    parser.add_argument('--batchSize',default=16,type=int,help='batchsize for both test/train')
    parser.add_argument('--maxEpochs',default=50,type=int,help='maximum epochs')
    parser.add_argument('--modelfile',default=None,help='Model Weight file path')
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
        wandb.init(project="[CLF] D4 Molecular Property Prediction",id=args.wid,resume='allow',config=args)
        print('WandB login completed ..')

    except Exception as ex : 
        print('Folder Creation Error , Exiting : {}'.format(ex))
        sys.exit()

 
class Trainer(object):

    def __init__(self, args):
        self.args=args 

        # Training Parameters 
        self.batchSize = args.batchSize
        self.ntasks=args.ntasks
        self.lr = args.learningRate
        self.maxEpochs=args.maxEpochs
        self.testCheckpointPath = args.modelfile
        self.expdir = args.expdir

        # Create folders 
        create_folder(args)

        # Model Initialisation 
        self.model=D4MP()
        self.model.to(device)

        # Specify the data csv files  
        train_df = pd.read_csv('./data/reg_train.csv')
        valid_df = pd.read_csv('./data/reg_test.csv')

        # Converting Dataframes to Dataclass 
        train_dataset = Dataclass(train_df)
        valid_dataset = Dataclass(valid_df)

        # DataLoaders 
        self.train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=self.batchSize, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=self.batchSize,shuffle=True)

        # Optimisers 
        self.optimizer =torch.optim.Adam(
            self.model.parameters(),
            lr=args.learningRate,
            amsgrad=False)
        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer,step_size=5,gamma=0.1)

        # Loss function 
        self.reg_loss_fn=torch.nn.MSELoss(reduce=False)

    def resume(self, path):
        print(' Model Training Resumed .. !')
        self.model_path=path
        save_state = torch.load(self.model_path)
        new_state=self.modified_state_dict(save_state)
        try :
            self.model.load_state_dict(new_state)
            print('Model loaded ...  ')
        except Exception as ex :
            print('Model loading error ___ {}'.format(ex)) 
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        print('Model reloaded to resume from Epoch %d' % (self.epoch))

    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'd4m_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }
        save_name = os.path.join(self.args.expdir, 'checkpoints', 'epoch_%d.pth'%(epoch))
        torch.save(save_state, save_name)
        print('Saved model @ Epoch : {}'.format(self.epoch))

    def loop(self):
        for epoch in range(0,self.maxEpochs):
            self.epoch = epoch
            self.lr_decay.step()
            wandb.log({'current_lr':self.optimizer.param_groups[0]['lr'],'epoch':self.epoch})
            self.train(epoch)
            print('Epoch : {} ---- Instances Length : {} '.format(self.epoch,len(self.train_loader)))


    def train(self,epoch):
        print('Starting training...')
        self.model.train()
        wandb.watch(self.model,criterion=torch.nn.BCELoss(reduce=False),log="all")
        countErr=0
        running_loss = []
        print('Train Epoch : {}'.format(epoch))
        for step,samples in enumerate(self.train_loader):
            # try:
                drug_graphs=samples[0]
                taskMasks=torch.tensor((samples[1])).reshape((self.batchSize,1,self.ntasks)).to(device).float()
                labels=torch.tensor(samples[2]).to(device).float()
                y_out= self.model([drug_graphs])
                y_out=y_out.reshape((self.batchSize,1,self.ntasks))
                reg_loss=self.reg_loss_fn(y_out,labels).reshape((self.batchSize,self.ntasks,1))
                loss = torch.bmm(taskMasks,reg_loss)
                loss= torch.mean(loss,dim=0)
                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.cpu().detach().numpy())
                # except Exception as exp : 
                #     countErr=countErr+1
                #     print('TrainException : {} Errors : {}'.format(exp,countErr))
                #     continue
                

        # Validation 
        metrics_dict,val_loss=self.validate(epoch)

        print('\n')
        print('Epoch : {}'.format(epoch))
        print("TRAIN - Epoch: {}  Training loss:{} ".format(epoch,(np.mean(np.array(running_loss)))))
        print("VALIDATE - Epoch: {} , Validation_Loss : {} ".format(epoch,val_loss))
        print('\n')


        # WandB Logging 
        wandb.log({'avg_train_error':np.mean(np.array(running_loss)),'epoch':epoch})
        wandb.log({'learning_rate':self.optimizer.param_groups[0]['lr'],'epoch':epoch})
        wandb.log({'train_data_errors':countErr,'epoch':epoch})
        wandb.log({'avg_validation_error':val_loss,'epoch':epoch})
        for i in range(self.ntasks):
            for key in metrics_dict['task_'+str(i)].keys():
                wandb.log({'task{}_{}'.format(i,key):metrics_dict['task_'+str(i)][key],'epoch':epoch})

        # Save model checkpoint ..
        self.save_checkpoint(self.epoch)

    
    def validate(self,epoch):
            # Set the model to evaluation model 
            errorPoints=0
            total_loss=0.0
            epoch_loss=[]
            tasks=[]
            # Metrics holders for all 
            metrics=['mse']
            for i in range(self.ntasks):
                task={'mse':[]}
                tasks.append(task)

            # Iterating through samples .
            self.model.eval() 
            for step,samples in enumerate(self.valid_loader):
                try:
                    drug_graphs=samples[0]
                    taskMasks=torch.tensor((samples[1])).reshape((self.batchSize,1,self.ntasks)).to(device).float()
                    labels=torch.tensor(samples[2]).to(device).float()
                    y_out= self.model([drug_graphs])
                    y_out=y_out.reshape((self.batchSize,1,self.ntasks))
                    reg_loss=self.reg_loss_fn(y_out,labels).reshape((self.batchSize,self.ntasks,1))
                    loss = torch.bmm(taskMasks,reg_loss)
                    loss= torch.mean(loss,dim=0)
                    epoch_loss.append(loss.cpu().detach().numpy())

                    # Metrics 
                    for i,task in enumerate(tasks):
                        # taskmask tho malli we need to multiply or do only if taskMask = 1 
                        y_out_i=y_out[:,:,i].reshape((self.batchSize,1,1)).float()
                        y_gnd_i=labels[:,:,i].reshape((self.batchSize,1,1)).float()
                        
                        task_mask_i=taskMasks[:,:,i].reshape((self.batchSize,1,1)).float()

                        # Multiply via task_mask_i before computing MSE 
                        y_out_i=torch.multiply(y_out_i,task_mask_i)
                        y_gnd_i=torch.multiply(y_gnd_i,task_mask_i)
                        mse_loss_i = mean_squared_error(y_out_i,y_gnd_i)
                    
                        # Metrics
                        task['mse'].append(mse_loss_i.cpu().detach().numpy())

                except Exception as exp : 
                    errorPoints+=1
                    continue

            # Final Metrics
            metrics=['mse']
            finalDict={}
            for i,task in enumerate(tasks):
                finalDict['task_'+str(i)]={}
                for metric in metrics :
                    finalDict['task_'+str(i)][metric]=np.mean(np.asarray(task[metric],dtype=np.float))
            
            # Net loss 
            net_loss = np.mean(np.array(epoch_loss,dtype=np.float).flatten())
            # WandB logging 
            wandb.log({'validation_data_errors':errorPoints,'epoch':epoch})
            
            return(finalDict,net_loss)


if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    if(args.mode=='train'):
        trainer.loop()
    else:
        trainer.test()
