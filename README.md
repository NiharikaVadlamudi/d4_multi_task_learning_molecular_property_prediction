# Multi Task Learning for Molecular Property Prediction

Description : This repository will consist baseline experiments for Multi-Task Molecular Property Prediction on Harvard's TDC dataset . 

## Installation
Easy installation via [conda](https://www.anaconda.com/) : 
```bash
conda env create --file d4_mtp_env.yml --python=3.8 
conda activate d4_mtp_env
```
## TDC Regression Tasks 
A total of 6 tasks - Caco-2 , Lipophilicity , Solubility (AqSolDB) , PPBR , Acute Toxicity LD50 & Clearance (Hepatocyt) are catergorised under regression. To generate the datafiles ( train & test ) , run the following commands : 
```bash
cd tdc_regression
bash reg_data_generation.sh 
```
For training the network : 
```bash
cd tdc_regression
bash train_reg_model.sh 
```
## Weights & Biases Page 
A one-stop portal for observing current experiments :  
[MTP Experiments ] (https://wandb.ai/amber1121/D4%20Molecular%20Property%20Prediction?workspace=user-amber1121) 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


