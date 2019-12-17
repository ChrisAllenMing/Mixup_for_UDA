from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import os, sys
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import trainer_mixed
import trainer_mixed_old

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to source dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='pretrained model')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='Number of filters to use in the generator network')
    parser.add_argument('--ndf', type=int, default=64, help='Number of filters to use in the discriminator network')
    parser.add_argument('--nepochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for CPU training')
    parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
    parser.add_argument('--method', default='DM-UDA', help='Method to train| DM-UDA, sourceonly')
    parser.add_argument('--manualSeed', type=int, default = 400, help='manual seed')
    parser.add_argument('--KL_weight', type=float, default = 1.0, help='weight for KL divergence')
    parser.add_argument('--adv_weight', type=float, default = 0.1, help='weight for adv loss')
    parser.add_argument('--lrd', type=float, default=0.0001, help='learning rate decay, default=0.0001')
    parser.add_argument('--gamma', type=float, default = 0.3, help='multiplicative factor for target adv. loss')
    parser.add_argument('--delta', type=float, default = 0.3, help='multiplicative factor for mix adv. loss')
    parser.add_argument('--source_dataset', default='svhn', help='name of the source dataset')
    parser.add_argument('--target_dataset', default='mnist', help='name of the target dataset')
    parser.add_argument('--alpha', type=float, default = 2.0, help='the hyperparameter for beta distribution')
    parser.add_argument('--clip_thr', type = float, default = 0.1, help='the threshold of mixup ratio clipping')

    opt = parser.parse_args()
    print(opt)

    # Creating log directory
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'source_generation'))
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'target_generation'))
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'models'))
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'mix_images'))
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'mix_generation'))
    except OSError:
        pass

    # Setting random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.gpu>=0:
        torch.cuda.manual_seed_all(opt.manualSeed)

    # GPU/CPU flags
    cudnn.benchmark = True
    if torch.cuda.is_available() and opt.gpu == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --gpu [gpu id]")
    if opt.gpu>=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    # Creating data loaders
    mean = np.array([0.44, 0.44, 0.44])
    std = np.array([0.19, 0.19, 0.19])

    # define the directory for train and validation
    source_train_root = os.path.join(opt.dataroot, opt.source_dataset, 'trainset')
    source_val_root = os.path.join(opt.dataroot, opt.source_dataset, 'testset')
    target_train_root = os.path.join(opt.dataroot, opt.target_dataset, 'trainset')
    target_val_root = os.path.join(opt.dataroot, opt.target_dataset, 'testset')

    # define the preprocess operation
    resize_shape = (opt.imageSize, opt.imageSize)
    transform_source = transforms.Compose(
        [transforms.Resize(resize_shape), transforms.ToTensor(), transforms.Normalize(mean, std)])
    transform_target = transforms.Compose(
        [transforms.Resize(resize_shape), transforms.ToTensor(), transforms.Normalize(mean, std)])

    # define dataloaders
    source_train = dset.ImageFolder(root=source_train_root, transform=transform_source)
    source_val = dset.ImageFolder(root=source_val_root, transform=transform_source)
    target_train = dset.ImageFolder(root=target_train_root, transform=transform_target)
    target_val = dset.ImageFolder(root=target_val_root, transform=transform_target)

    source_trainloader = torch.utils.data.DataLoader(source_train, batch_size=opt.batchSize, shuffle=True,
                                                     num_workers=opt.workers, drop_last=True)
    source_valloader = torch.utils.data.DataLoader(source_val, batch_size=opt.batchSize, shuffle=False,
                                                   num_workers=opt.workers, drop_last=False)
    target_trainloader = torch.utils.data.DataLoader(target_train, batch_size=opt.batchSize, shuffle=True,
                                                     num_workers=opt.workers, drop_last=True)
    target_valloader = torch.utils.data.DataLoader(target_val, batch_size=opt.batchSize, shuffle=False,
                                                     num_workers=opt.workers, drop_last=False)

    nclasses = len(source_train.classes)
    
    # Training
    if opt.method == 'DM-UDA':
        DM_UDA_trainer = trainer_mixed_old.DM_UDA(opt, nclasses, mean, std, source_trainloader,
                                                  source_valloader, target_trainloader, target_valloader)
        DM_UDA_trainer.train()
    elif opt.method == 'sourceonly':
        sourceonly_trainer = trainer_mixed_old.Sourceonly(opt, nclasses, source_trainloader, target_valloader)
        sourceonly_trainer.train()
    else:
        raise ValueError('method argument should be DM-UDA or sourceonly')

if __name__ == '__main__':
    main()

