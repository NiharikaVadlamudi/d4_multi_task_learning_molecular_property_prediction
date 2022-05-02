'''
Dataloader 
'''
# generic libraries 
import numpy as np 

# rdkit imports
from rdkit import RDLogger
from rdkit import rdBase
from rdkit import Chem


# torch imports 
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset

# dgl imports 
import dgl 

# file imports 
from utils import * 
from molecular_graph import * 

# We have regression or binary classification cases only 
reg_loss_fn = torch.nn.MSELoss()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
device='cpu'


# collate function 
def collate(samples):
    drug_graphs,taskMasks,labels = map(list, zip(*samples))
    drug_graphs = dgl.batch(drug_graphs)
    return drug_graphs,taskMasks,labels

def one_hot(a, num_classes=2):
    a=np.asarray(a,dtype=np.int)
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

class Dataclass(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    # Dataframe format 
    def __getitem__(self,idx): 
        self.dataset=self.dataset.dropna(inplace = False)
        y_columns= [ col  for col in list(self.dataset.columns) if 'y_' in col]
        drug = self.dataset.loc[idx]['molecule_smiles']
        taskMask=self.dataset.loc[idx]['taskMask']
        taskMask=converttoList(taskMask)
        taskMask=np.asarray(taskMask,dtype=np.float)
        currtasks=np.count_nonzero(taskMask==1.0)
        taskMask=(1/currtasks)*taskMask

        # Processing on the molecule 
        mol = Chem.MolFromSmiles(drug)
        mol = Chem.AddHs(mol)
        drug= Chem.MolToSmiles(mol)
        drug_graph = get_graph_from_smile(drug)

        # Create a y vector 
        y_gd=[]
        for i,y_label in enumerate(y_columns):
            y_gd.append(self.dataset.loc[idx][y_label])
        y = np.asarray(y_gd).astype(np.float)
        
        return [drug_graph,taskMask,[y]]