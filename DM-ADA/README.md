# Adversarial Domain Adaptation with Domain Mixup

This is the implementation of proposed DM-ADA method.

## Requirements

* Python 2.7
* PyTorch 0.4.0 / 0.4.1

## Prerequisites

Download MNIST, SVHN and USPS datasets, and prepare the datasets with following structure:
```
/Dataset_Root
 └── mnist
     ├── trainset
     │   ├── subfolders for 0 ~ 9
     ├── testset
 ├── svhn
 ├── usps
```

## Training

* Train the Source-only baseline (validation on target domain is conducted for each epoch): 
```
python main.py --dataroot <Datset_Root> --method sourceonly --source_dataset <Source_Domain> --target_dataset <Target_Domain> 
```

* Train the DM-ADA model (validation on target domain is conducted for each epoch): 
```
python main.py --dataroot <Datset_Root> --method DM-ADA --source_dataset <Source_Domain> --target_dataset <Target_Domain> 
```
