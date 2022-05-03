'''
Pooling Datasets across the TDC prediction tasks 
Supports tasks from ADME group 
'''

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

# Rdkit imports 
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit import RDLogger

# tdc imports
import tdc
from tdc import utils

# global parameters - all binary cls first & then regression cls
propertyList=['bioavailability_ma','cyp2d6_veith','ames','herg','caco2_wang','lipophilicity_astrazeneca','solubility_aqsoldb','ppbr_az','ld50_zhu','clearance_hepatocyte_az']
defaultClfList=['bioavailability_ma','cyp2d6_veith','ames','herg']
defaultRegList=['caco2_wang','lipophilicity_astrazeneca','solubility_aqsoldb','ppbr_az','ld50_zhu','clearance_hepatocyte_az']

# Parser 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasktype',default='clf',help='choose among the options reg/clf/hybrid')
    parser.add_argument('--props', default=None,help="mention  the required molecular properties (str list) ",nargs='+')
    parser.add_argument('--mode',default='train',help='benchmark train / test split')
    parser.add_argument('--csvWrite',default=True)
    parser.add_argument('--fileName',default='train_combined',help='Output file name')
    args = parser.parse_args()
    return args

# Utils

def convertSMILES(smiles):
    '''
    Removes all canonical versions of molecules; need to apply this on every data pool of TDC , to maintain 
    consistency . 
    '''
    return(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))

def intersectionSet(list1,list2):
    '''
    Checks for common elements and returns them 
    '''
    a_set = set(list1)
    b_set = set(list2)
    c=(a_set.intersection(b_set))
    return(list(c))

def molecularDictionary(dframe,dictformat=False):
    '''
    Molecule names will be a part of the dictionary ; all the remaining columns 
    will act as values ( no heirarchy ) 
    Note : Set canonicalized 'DRUG' SMILE as the index of the dataframe .
    '''
    # Check if the dictionary column is unique in nature 
    # dframe=dframe.set_index('Drug')
    if(dictformat):
        return dframe.T.to_dict('dict')
    else:
        return dframe.T.to_dict('list')
    
    
# Data Normalisation (Regression Specific) 
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        if(str(column).find('y_')>0):
            print('Entered column ! : {}'.format(column))
            df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    return df_std
    

# Main Functions 

def datapreprocessing(name,group='ADMET'): 
    '''
    Returns the processed benchmark dataframe for use given 
    the selected combination of group and sub-task dataset name.
    '''
    # Fetch the family 
    if(group.lower()=='admet'):      
        from tdc.benchmark_group import admet_group
        groupData = admet_group()
        names = utils.retrieve_benchmark_names('ADMET_Group')
        if(name in names):
            # Fetch the benchmarck splits directly 
            benchmark = groupData.get(name)
            train_val, test = benchmark['train_val'], benchmark['test']
        else:
            print('Non Existant sub-group name in {}'.format(group))
            return None 
    # Clean-up the dataframes for furthur usage
    dataframes=[train_val,test]
    resdfs=[]
    for df in dataframes:
        df=pd.DataFrame(df)
        df=df.dropna(inplace = False)
        df=df.drop('Drug_ID',axis=1,inplace=False)
        df['Drug']=df['Drug'].apply(convertSMILES)
        df=z_score(df)
        df=df.set_index('Drug')
        df=df.rename(columns = {'Y':'y_{}'.format(name[0:3].lower())},inplace = False)
        df['family']=str(group)
        resdfs.append(df)
    return(resdfs[0],resdfs[1])


def combiningData(args):
    '''
    Input : Set of dataframes(train,test)(indexed by 'Drug') that you want to combine.
    mode : train/test.
    csvWrite : For converting the dataset back to csv format.
    fileName : In case csvWrite is True , a valid filename must be provided.
    
    '''
    tasktype=args.tasktype
    props=args.props
    mode=args.mode
    csvWrite=args.csvWrite
    fileName=args.fileName
    propList=[]

    if(tasktype=='clf'):
        propList=defaultClfList.copy()
    if(tasktype=='reg'):
        propList=defaultRegList.copy()
    if(tasktype=='hybrid'):
        propList=propertyList.copy()
    if(args.props!=None):
         propList=args.props.copy()
        

    if(len(propList)==0):
        print('Empty array of property names , minimum needed is 1 ')
        return 

    if(mode=='train'):
        gindex=0
    elif(mode=='test'):
        gindex=1
    else:
        print('Invalid Mode,Exiting..')
        return None 
    
    # Declaration 
    listofdicts=[]
    mol_list=[]
    y_vals=[]
    listofdfs=[]

    for name in propList:
        out = datapreprocessing(name,group='ADMET')
        listofdfs.append(out)

    for d in listofdfs:
        y_vals.append(list(d[gindex].columns)[0])
        mol_dict = molecularDictionary(d[gindex],dictformat=True)
        mol_list += mol_dict.keys()
        listofdicts.append(mol_dict)

    # Get unique keys of dataset 
    mol_list=list(set(mol_list))
    
    # New combined dictionary 
    combinedDict={}
    for k in mol_list:
        combinedDict[k]={}
        combinedDict[k]['taskMask']=[]
        for y_label in y_vals:
            combinedDict[k][y_label]=None
        # Now , iterate through other dictionaries to update the value , if present .
        for ind,dct in enumerate(listofdicts):
            if k in dct.keys():
                combinedDict[k][y_vals[ind]]=dct[k][y_vals[ind]]
                combinedDict[k]['taskMask'].append(1)
            else:
                combinedDict[k][y_vals[ind]]=int(0)
                combinedDict[k]['taskMask'].append(0)

    # Dataframe conversion 
    if(csvWrite):
        assert(fileName!=None)
        res = pd.DataFrame.from_dict(combinedDict,orient ='index')
        res.index.rename('molecule_smiles', inplace=True)
        res=res.dropna(inplace = False)
        res.to_csv('./data/{}.csv'.format(fileName),header=True,index=True)
    return(combinedDict)


if __name__ == '__main__':
    args = get_args()
    res=combiningData(args)
    print('Datafile generated ...')
