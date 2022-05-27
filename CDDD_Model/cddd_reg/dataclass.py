''' 
Modified Dataclass for CDDD Decriptors Part 
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
device='cpu'


def one_hot(a, num_classes=2):
    a=np.asarray(a,dtype=np.int)
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def collate(batch):
    drug_graphs,taskMasks,labels = map(list, zip(*batch))
    drug_graphs=torch.tensor(drug_graphs).to(device).to(torch.float)
    taskMasks=torch.tensor(taskMasks).to(device).to(torch.float)
    labels=torch.tensor(labels).to(device).to(torch.float)
    print('batch cheseam bro')
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

    # collate function 

    def fetch(self,idx):
        try : 
            y_desc=[]
            for i,col in enumerate(self.y_desc_names):
                y_desc.append(self.dataset.loc[idx][col])
            y_desc=np.asarray(y_desc,dtype=np.float)
            taskMask=self.dataset.loc[idx]['taskMask']
            taskMask=converttoList(taskMask)
            taskMask=np.asarray(taskMask,dtype=np.float)
            currtasks=np.count_nonzero(taskMask==1.0)
            taskMask=(1/currtasks)*taskMask
            # No need of graph descriptors as well .
            # Create a y vector 
            y_gd=[]
            for i,y_label in enumerate(self.y_columns):
                y_gd.append(self.dataset.loc[idx][y_label])

            y = np.asarray(y_gd).astype(np.float) 
        
        except Exception as exp : 
            print('Dataloading Errror : {}'.format(exp))
            return None 
        
        return [y_desc,taskMask,[y]]

    # Dataframe format 
    def __getitem__(self,idx):
        opt = self.fetch(idx)
        if(opt is None ):
            res=None
            found=False
            while(found==False):
                new_idx = np.random.randint(0,len(self.dataset)-1)
                res=self.fetch(new_idx)
                if(res==None):
                    continue
                [y_desc,taskMask,[y]]=res
                found=True
        else:
            [y_desc,taskMask,[y]]=opt
            
        return [y_desc,taskMask,[y]]






































































# ''' 
# Modified Dataclass for CDDD Decriptors Part 
# '''

# # generic libraries 
# from tkinter import E, Y
# import numpy as np 

# # rdkit imports
# from rdkit import RDLogger
# from rdkit import rdBase
# from rdkit import Chem

# # dgl imports 
# import dgl 
# from dgl import *

# # torch imports 
# import torch
# import torch.nn as nn 
# from torch.utils.data import DataLoader, Dataset

# # file imports 
# from utils import * 


# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# device='cpu'


# # collate function 
# def collate(samples):
#     drug_graphs,taskMask,labels = map(list, zip(*samples))
#     drug_graphs = list(filter(None,  drug_graphs))
#     taskMask= list(filter(None, taskMask))
#     labels = list(filter(None, labels))
    
#     print('cOLLATE : {} {} {} '.format(type(drug_graphs),type(taskMask),type(labels)))

#     drug_graphs = dgl.batch(drug_graphs)
#     return drug_graphs,taskMask,labels

# def one_hot(a, num_classes=2):
#     a=np.asarray(a,dtype=np.int)
#     return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

# class Dataclass(Dataset):
#     def __init__(self, dataset,lenDescriptors=512):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)
    
#     # Dataframe format 
#     def __getitem__(self,idx):
#         try : 
#             self.dataset=self.dataset.dropna(inplace = False)
#             y_desc_names=[]
#             y_desc_names =[col  for col in list(self.dataset.columns) if 'cddd_' in col]
#             y_columns =[col  for col in list(self.dataset.columns) if 'y_' in col]
#             y_desc=[]
#             for i,col in enumerate(y_desc_names):
#                 y_desc.append(self.dataset.loc[idx][col])
            
#             y_desc=np.asarray(y_desc,dtype=np.float)
#             taskMask=self.dataset.loc[idx]['taskMask']
#             taskMask=converttoList(taskMask)
#             taskMask=np.asarray(taskMask,dtype=np.float)
#             currtasks=np.count_nonzero(taskMask==1.0)
#             taskMask=(1/currtasks)*taskMask
#             # No need of graph descriptors as well .
#             # Create a y vector 
#             y_gd=[]
#             for i,y_label in enumerate(y_columns):
#                 y_gd.append(self.dataset.loc[idx][y_label])
#             y = np.asarray(y_gd).astype(np.float) 
#             print('Y_desc : {}'.format(y_desc.shape))
#             print('Y_cols : {}'.format(y.shape))
        
#         except Exception as exp : 
#             print('Dataloading Errror : {}'.format(exp))
#             return [None,None,[None]]

#         return [y_desc,taskMask,[y]]







