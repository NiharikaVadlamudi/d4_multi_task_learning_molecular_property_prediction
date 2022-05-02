'''
D4MP Model Defination - Regression
Motivation : 
    * N-Task Attention Head & GIN Architecture 
        - GATLayers with multiple heads (N-tasks)
        - GINLayers : SOTA for node embeddings 
        - Each task has it's own moduleBlock depending on Regression / Classification Task
'''

import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set,GATConv,GINConv

# torch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F

# generic import
import numpy as np
device='cpu'

class D4MP(nn.Module):
    '''
    Baseline class for multi-task property prediction 
    Supports both regression & binary classification task , 
    defined by a prior ordered sub-task list.
    '''
    def __init__(self,
                 ntasks=6,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_set2_set=2,
                 num_layer_set2set=1,
                 intermediate_dim=32,noclf=0,noreg=6):

        super(D4MP, self).__init__()

        # Generic params 
        self.ntasks=ntasks
        self.intermediate_dim=intermediate_dim

        # Input Graph Features 
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim

        # Edge Features
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim

        # Set2Set Layers 
        self.num_step_set2set=num_step_set2_set
        self.num_layer_set2set=num_layer_set2set

        # Layers Initialisation 
        self.gat=dgl.nn.pytorch.conv.GATConv(self.node_hidden_dim,self.node_hidden_dim,num_heads=self.ntasks,allow_zero_in_degree=True).to(device)
        # self.gat=dgl.nn.pytorch.conv.GATv2Conv(self.node_hidden_dim,self.node_hidden_dim,num_heads=self.ntasks,allow_zero_in_degree=True,feat_drop=0.3,attn_drop=0.4,bias=True).to(device)
        self.gin_linear_layer =nn.Linear(self.node_input_dim,self.node_hidden_dim).to(device)
        self.gin = dgl.nn.pytorch.conv.GINConv(self.gin_linear_layer,aggregator_type='mean').to(device)
        
        self.set2set = Set2Set(2*node_hidden_dim, self.num_step_set2set,self.num_layer_set2set).to(device)

        # Ordered List --- from our prior knowledge on the ordering of property prediction tasks
        self.task_layers = nn.ModuleList()
        for i in range(0,self.ntasks):
            self.task_layers.append(nn.Sequential( 
            nn.Linear(4 * self.node_input_dim, self.intermediate_dim).to(device),
            nn.ELU(),
            nn.Linear(self.intermediate_dim,int(self.intermediate_dim/2)).to(device),
            nn.ELU(),
            nn.Linear(int(self.intermediate_dim/2),1).to(device)))

    def forward(self,data):
        drug = data[0]
        # First set of node features from GINLayer 
        drug_features = self.gin(drug,drug.ndata['x'].float())
        # Attention-wise features 
        attn_features = self.gat(drug,drug.ndata['x'].float())
        # Concatenate the output features 
        for i,layer in enumerate(self.task_layers):
            attn_head = attn_features[:,i,:]
            z = torch.cat((drug_features,attn_head),dim=1)
            z = self.set2set(drug,z)
            z = layer(z)
            if(i==0):
                z_out=z
            else:
                z_out=torch.cat((z_out,z))
        return(z_out)