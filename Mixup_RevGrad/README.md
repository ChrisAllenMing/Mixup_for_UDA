# Adversarial Domain Adaptation with Domain Mixup

This is the implementation of Domain Mixup strategy on a classifical adversarial domain adaptation method, RevGrad.

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

* Train the original RevGrad model (validation on target domain is conducted for each epoch): 
```
python RevGrad.py --root_path <Datset_Root> --source <Source_Domain> --target <Target_Domain> 
```

* Train the RevGrad model with Domain Mixup strategy (validation on target domain is conducted for each epoch): 
```
python RevGrad_mixup.py --root_path <Datset_Root> --source <Source_Domain> --target <Target_Domain> 
```
