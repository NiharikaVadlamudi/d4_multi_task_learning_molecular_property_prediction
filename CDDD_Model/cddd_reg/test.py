# Import libraries
import csv 
import sys
import csv 
import json
from random import shuffle
import numpy as np

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
from torchmetrics.functional import accuracy,precision,recall,auc,mean_squared_error,spearman_corrcoef,mean_absolute_error
from torch.utils.data import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

# File Imports 
from network import *
from dataclass import * 


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

taskList = ['caco','lipophilicity','aqsol','ppbr','tox','clearance']


# Parser 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name',default='test_reg_tdc_task',help='name of the project')
    parser.add_argument('--wid',default=wandb.util.generate_id(),help='wid initialisation')
    parser.add_argument('--expdir',default='tdc_reg',help='experiment folder')
    parser.add_argument('--mode',default='test',help='train/evaluate the model')
    parser.add_argument('--ntasks',default=6,type=int,help='N Classification Tasks')
    parser.add_argument('--batchSize',default=1,type=int,help='batchsize for both test/train')
    parser.add_argument('--modelfile',default=None,type=str,help='Model Weight file path')
    args = parser.parse_args()
    return args


# Test Folder Creation 
def create_folder(args):
    try : 
        path=args.expdir
        pathResults=args.expdir+'/test/results/'
        if(args.mode=='test'):
            os.system('mkdir -p %s' % pathResults)
        print('Experiment folder created ')

        # WandB Initialisation 
        WANDB_API_KEY="4169d431880a3dd9883605d90d1fe23a248ea0c5"
        WANDB_ENTITY="amber1121"
        os.environ["WANDB_RESUME"] ="allow"
        os.environ["WANDB_RUN_ID"] = args.wid
        wandb.init(project="[TEST-Reg] D4 Molecular Property Prediction",id=args.wid,resume='allow',config=args)
        print('WandB login completed ..')

    except Exception as ex : 
        print('Folder Creation Error , Exiting : {}'.format(ex))
        sys.exit()

class Tester(object):
    def __init__(self, args):
        self.args=args 

        # Training Parameters 
        self.batchSize = args.batchSize
        self.ntasks=args.ntasks
        self.testCheckpointPath = args.modelfile
        self.expdir = args.expdir

        # Create folders 
        create_folder(args)

        # Model Initialisation -> Testing 
        self.model=CDDD_FC()
        #self.model=ANYA()
        self.model.to(device)
        self.loadModel()
        self.model.eval()

        # Specify the data csv files 
        test_df = pd.read_csv('./data/reg_test.csv')

        # Converting Dataframes to Dataclass 
        test_dataset = Dataclass(test_df)

        # DataLoaders 
        self.test_loader = DataLoader(test_dataset, collate_fn=collate, batch_size=self.batchSize, shuffle=True)
        self.reg_loss_fn=torch.nn.MSELoss(reduce=False)


        # Load model checkpoint 
    def loadModel(self):
        self.model_path=self.testCheckpointPath
        try :
            self.model.load_state_dict(torch.load(self.model_path)["d4m_state_dict"])
            print('Model loaded ...  ')
        except Exception as ex :
            print('Model loading error ___ {}'.format(ex)) 


    # Testing the model .. 
    def test(self):
        taskErrors = []
        tasks=[]
        metrics=['mae','spcoeff']
        for i in range(self.ntasks):
            task={'mae':[],'spcoeff':[]}
            tasks.append(task)
        
        task_coco={'y_gnd':[],'y_pred':[]}
        task_lip={'y_gnd':[],'y_pred':[]}
        task_aqsol={'y_gnd':[],'y_pred':[]}
        task_ppbr={'y_gnd':[],'y_pred':[]}
        task_tox={'y_gnd':[],'y_pred':[]}
        task_clearance={'y_gnd':[],'y_pred':[]}


        taskDicts = [task_coco,task_lip,task_aqsol,task_ppbr,task_tox,task_clearance] 

        for step,samples in enumerate(self.test_loader):
            print('Step : {}'.format(step))
            try:
                drug_graphs=samples[0]
                taskMasks=torch.tensor((samples[1])).reshape((self.batchSize,1,self.ntasks)).to(device).float()
                labels=torch.tensor(samples[2]).to(device).float()
                y_out= self.model([drug_graphs])
                y_out=y_out.reshape((self.batchSize,1,self.ntasks))
                reg_loss=self.reg_loss_fn(y_out,labels).reshape((self.batchSize,self.ntasks,1))
                loss = torch.bmm(taskMasks,reg_loss)
                loss= torch.mean(loss,dim=0)

                # Metrics 
                for i,task in enumerate(tasks):
                    # taskmask tho malli we need to multiply or do only if taskMask = 1 
                    y_out_i=y_out[:,:,i].reshape((self.batchSize,1)).float()
                    y_gnd_i=labels[:,:,i].reshape((self.batchSize,1)).float()
                    task_mask_i=taskMasks[:,:,i].reshape((self.batchSize,1)).float()

                    # Multiply via task_mask_i before computing MAE
                    y_out_i=y_out_i.cpu().detach().numpy().tolist()[0][0]
                    y_gnd_i=y_gnd_i.cpu().detach().numpy().tolist()[0][0]
                    task_mask_i=task_mask_i.cpu().detach().numpy().tolist()[0][0]

                    if(int(task_mask_i)!=0):
                        taskDicts[i]['y_gnd'].append(y_gnd_i)
                        taskDicts[i]['y_pred'].append(y_out_i)
            
            except Exception as exp : 
                print('Exp : {}'.format(exp))
                continue
                
        
        # Task Dicts 
        maes=[]
        spcs=[]
        for j,t in enumerate(taskDicts):
            num = len(list(t.keys()))
            print('Task Length : {}'.format(num))
            y_gd_i =torch.from_numpy(np.asarray(list(t['y_gnd']),dtype=np.float))
            y_pred_i=torch.from_numpy(np.asarray(list(t['y_pred']),dtype=np.float))
            maes.append((1/num)*mean_absolute_error(y_gd_i,y_pred_i))
            if(j==5):
                spcs.append((1/num)*spearman_corrcoef(y_gd_i,y_pred_i))

        # Logging the MAES & Spearman Coeffs 

        for j,t in enumerate(taskList):
            wandb.log({'task_{}_mae'.format(taskList[j]): maes[j]})
            if(j==5):
                wandb.log({'task_{}_spc'.format(taskList[j]):spcs[0]})

        # Printing values 
        mae_dict = zip(taskList,maes)
        spc_dict= zip(taskList,spcs)

        print('MAE is : {}'.format(mae_dict))
        print('SPC is : {}'.format(spc_dict))



if __name__ == '__main__':
    args = get_args()
    tester = Tester(args)
    if(args.mode=='test'):
        tester.test()
