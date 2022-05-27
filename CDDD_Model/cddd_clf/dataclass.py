''' 

Modified  Classicification Dataclass for CDDD Decriptors Part 

'''

# generic libraries 
import numpy as np 

# rdkit imports
from rdkit import RDLogger
from rdkit import rdBase
from rdkit import Chem

# dgl imports 
import dgl 
from dgl import *
from dgl import DGLGraph

# torch imports 
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset

# file imports 
from utils import * 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# Comment it out , to increase the speed .
device='cpu'



def one_hot(a, num_classes=2):
    a=np.asarray(a,dtype=np.int)
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def collate(batch):
    drug_graphs,taskMasks,labels = map(list, zip(*batch))
    drug_graphs = np.asarray(drug_graphs)
    drug_graphs=torch.from_numpy(drug_graphs).to(device)
    taskMasks=torch.tensor(taskMasks).to(device)
    labels=torch.tensor(labels).to(device)
    return drug_graphs,taskMasks,labels


class Dataclass(Dataset):

    def __init__(self, dataset,lenDescriptors=512):
        self.dataset = dataset
        self.lengthofDescriptors=lenDescriptors
        self.dataset=self.dataset.dropna(inplace = False)
        self.y_desc_names = [col  for col in list(self.dataset.columns) if 'cddd_' in col]
        self.y_columns =[col  for col in list(self.dataset.columns) if 'y_' in col]
    
    def __len__(self):
        return len(self.dataset)
    
    # Adding a fetch function to avoid errors at collate function 
    def fetch(self,idx):
        try : 
            y_desc=self.dataset.loc[idx,self.y_desc_names].tolist()
            y_desc=np.asarray(y_desc,dtype=np.float)
            # y_desc=torch.from_numpy(y_desc).to(torch.float).to(device)
            taskMask=self.dataset.loc[idx]['taskMask']
            taskMask=converttoList(taskMask)
            taskMask=np.asarray(taskMask,dtype=np.float)
            currtasks=np.count_nonzero(taskMask==1.0)
            # taskMask=torch.from_numpy(taskMask).to(torch.bool).to(device)
            y_gd = self.dataset.loc[idx,self.y_columns].tolist()
            y=np.asarray(y_gd).astype(np.float) 
            # y =torch.from_numpy(y).to(torch.float).to(device)

        except Exception as exp : 
            # print('EmptyRow : {} '.format(idx))
            return None
        
        return [y_desc,taskMask,[y]]

    def __getitem__(self,idx):
        opt = self.fetch(idx)
        if(opt is None):
            res=None
            found=False
            while(found is False):
                new_idx = np.random.randint(0,len(self.dataset)-1)
                res=self.fetch(new_idx)
                if(res==None):
                    continue
                [y_desc,taskMask,[y]]=res
                found=True
        else:
            [y_desc,taskMask,[y]]=opt
        return [y_desc,taskMask,[y]]
